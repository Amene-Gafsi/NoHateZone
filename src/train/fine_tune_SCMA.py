import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import LinearLR

from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, roc_curve
)
from sklearn.model_selection import GroupKFold

import pandas as pd
from pathlib import Path
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# =============================================================================
# Dataset + Model Definitions
# =============================================================================

class EmbeddingPairDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        return (
            idx,
            torch.tensor(row['img_embedding'], dtype=torch.float),
            torch.tensor(row['text_embedding'], dtype=torch.float),
            torch.tensor(row['label'], dtype=torch.float),
        )


def collate_fn(batch):
    idxs, imgs, txts, labels = zip(*batch)
    # pad image sequences
    imgs = [i if i.ndim==2 else i.unsqueeze(0) for i in imgs]
    P = max(i.shape[0] for i in imgs)
    D_img = imgs[0].shape[1]
    imgs_p = [torch.cat([i, torch.zeros(P-i.shape[0],D_img)], dim=0) for i in imgs]
    # pad text + mask
    txts = [t if t.ndim==2 else t.unsqueeze(0) for t in txts]
    N = max(t.shape[0] for t in txts)
    D_txt = txts[0].shape[1]
    txt_p, masks = [], []
    for t in txts:
        L = t.shape[0]
        pad = torch.zeros(N-L, D_txt)
        txt_p.append(torch.cat([t, pad], dim=0))
        masks.append(torch.tensor([True]*L + [False]*(N-L)))
    return (
        list(idxs),
        torch.stack(txt_p),       # text embeddings padded
        torch.stack(masks),       # mask for text
        torch.stack(labels),      # labels
        torch.stack(imgs_p),      # image embeddings padded
    )

class CrossModalBlock(nn.Module):
    def __init__(self, dim, heads, dropout, ff_ratio, num_classes=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.log_tau = nn.Parameter(torch.zeros(heads))
        self.norm1 = nn.LayerNorm(dim)
        self.drop1 = nn.Dropout(dropout)
        hidden = int(dim * ff_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.drop2 = nn.Dropout(dropout)
        self.classifier = nn.Linear(dim, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.heads = heads

    def forward(self, img_emb, txt_emb, text_mask=None):
        B, P, D = img_emb.shape
        _, N, _ = txt_emb.shape

        # project for multihead attention
        qkv = nn.functional.linear(img_emb, self.attn.in_proj_weight, self.attn.in_proj_bias)
        q, _ = qkv.split([D, 2*D], dim=-1)
        kvt = nn.functional.linear(txt_emb,
                                   self.attn.in_proj_weight[D:],
                                   self.attn.in_proj_bias[D:])
        k, v = kvt.split([D, D], dim=-1)

        # reshape for heads
        h = self.heads
        d = D // h
        Q = q.view(B, P, h, d).transpose(1, 2)
        K = k.view(B, N, h, d).transpose(1, 2)
        V = v.view(B, N, h, d).transpose(1, 2)

        # scaled dot-product with learnable temperature
        scores = (torch.einsum('bhpd,bhqd->bhpq', Q, K) / math.sqrt(d)) \
                 / torch.exp(self.log_tau).view(1, h, 1, 1)
        if text_mask is not None:
            mask = ~text_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        attn_out = torch.einsum('bhpq,bhqd->bhpd', weights, V)
        attn_out = attn_out.transpose(1,2).contiguous().view(B, P, D)
        out = nn.functional.linear(attn_out,
                                   self.attn.out_proj.weight,
                                   self.attn.out_proj.bias)

        # residual + feed-forward
        x = self.norm1(img_emb + self.drop1(out))
        x = self.norm2(x + self.drop2(self.ffn(x)))
        logits = self.classifier(x).squeeze(-1)
        return x, weights, logits, self.sigmoid(logits)

# =============================================================================
# Training & Evaluation Utilities
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for _, txts, masks, labels, imgs in tqdm(loader, desc='Training', leave=False):
        imgs, txts, masks, labels = (
            imgs.to(device), txts.to(device),
            masks.to(device), labels.to(device)
        )
        optimizer.zero_grad()
        _, _, logits, _ = model(imgs, txts, text_mask=masks)
        loss = criterion(logits.view(-1), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def collect_probs_labels(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for _, txts, masks, lbls, imgs in loader:
            imgs, txts, masks = imgs.to(device), txts.to(device), masks.to(device)
            _, _, _, p = model(imgs, txts, text_mask=masks)
            probs.extend(p.cpu().numpy().ravel().tolist())
            labels.extend(lbls.numpy().ravel().tolist())
    return np.array(probs), np.array(labels)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
    }


def select_threshold(y_true, y_prob):
    thresholds = np.linspace(0, 1, 101)
    f1s = [(t, f1_score(y_true, (y_prob>=t).astype(int))) for t in thresholds]
    return max(f1s, key=lambda x: x[1])[0]

# =============================================================================
# Main: Fixed Split + Seed Loop + ROC Plotting
# =============================================================================
def main():
    # output
    out_dir = Path('cv_results_fixed_split')
    out_dir.mkdir(parents=True, exist_ok=True)

    # load data
    df = pd.read_pickle('./fine_tune_hatemm/embeddings_hatemm_mert.pkl')
    parts = df['frameID'].str.rsplit('_frame', n=1, expand=True)
    df['video_group'] = parts[0]

    # fixed test split (20% groups) with constant RNG
    unique_groups = df['video_group'].unique()
    rng = np.random.RandomState(42)
    test_size = int(0.2 * len(unique_groups))
    test_groups = list(rng.choice(unique_groups, size=test_size, replace=False))

    df_test = df[df['video_group'].isin(test_groups)].reset_index(drop=True)
    df_train = df[~df['video_group'].isin(test_groups)].reset_index(drop=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim = df['img_embedding'].iloc[0].shape[-1]
    seeds = [0, 1, 50, 100, 1000]
    k_folds = 5
    epochs = 20

    # DataLoader for test set (no randomness)
    test_loader = DataLoader(
        EmbeddingPairDataset(df_test),
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )

    # precompute cross-validation folds (deterministic)
    gkf = GroupKFold(n_splits=k_folds)
    folds = list(gkf.split(df_train, groups=df_train['video_group']))

    all_results = {}
    roc_data = {}

    for seed in seeds:
        print(f"\n=== Running Seed {seed} ===")
        # set seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # store seed metrics
        fold_metrics = []

        for fold_idx, (tr_idx, val_idx) in enumerate(folds):
            print(f"Seed {seed} | Fold {fold_idx+1}/{k_folds}")
            df_tr = df_train.iloc[tr_idx].reset_index(drop=True)
            df_val = df_train.iloc[val_idx].reset_index(drop=True)

            # class-balanced sampler with seed-specific generator
            counts = np.bincount(df_tr['label'].astype(int))
            weights = (1.0 / counts)[df_tr['label'].astype(int)]
            gen = torch.Generator()
            gen.manual_seed(seed)
            sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(weights),
                num_samples=len(df_tr),
                replacement=True,
                generator=gen
            )

            # model init
            model = CrossModalBlock(dim, heads=16, dropout=0.6, ff_ratio=2).to(device)
            checkpoint = torch.load('./fine_tune_hatemm/best_model_weights.pth', map_location=device)
            model.load_state_dict(checkpoint, strict=False)

            optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
            scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs)

            # loss with pos weight
            pos_w = torch.tensor(
                [(df_tr['label']==0).sum() / (df_tr['label']==1).sum()],
                dtype=torch.float,
                device=device
            )
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

            # loaders
            train_loader = DataLoader(
                EmbeddingPairDataset(df_tr),
                batch_size=16,
                sampler=sampler,
                collate_fn=collate_fn
            )
            val_loader = DataLoader(
                EmbeddingPairDataset(df_val),
                batch_size=16,
                shuffle=False,
                collate_fn=collate_fn
            )

            # train
            for epoch in range(1, epochs+1):
                loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
                scheduler.step()
                print(f" Epoch {epoch}/{epochs} | Loss: {loss:.4f}")

            # evaluate on validation
            val_probs, val_labels = collect_probs_labels(model, val_loader, device)
            thresh = select_threshold(val_labels, val_probs)

            # evaluate on test
            test_probs, test_labels = collect_probs_labels(model, test_loader, device)
            metrics = compute_metrics(test_labels, test_probs, threshold=thresh)
            fold_metrics.append(metrics)

        # average metrics over folds
        df_fold = pd.DataFrame(fold_metrics)
        all_results[seed] = df_fold.mean().to_dict()
        # store ROC data for this seed
        roc_data[seed] = (test_labels, test_probs)

    # save results
    df_results = pd.DataFrame.from_dict(all_results, orient='index')
    df_results.index.name = 'seed'
    df_results.to_csv(out_dir / 'test_metrics.csv')

    # plot ROC curves for each seed
    plt.figure()
    for seed, (labels, probs) in roc_data.items():
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.plot(fpr, tpr, label=f'Seed {seed}')
    plt.plot([0,1], [0,1], '--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SCMA: ROC Curves per Seed')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'roc_per_seed.png')
    plt.close()

if __name__ == '__main__':
    main()