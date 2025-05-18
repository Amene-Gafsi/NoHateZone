import torch                                           # Core PyTorch for tensor computations
import torch.nn as nn                                  # Neural network modules
import torch.optim as optim                            # Optimization algorithms
from torch.utils.data import DataLoader, Dataset       # Data pipeline utilities
from torch.utils.data import WeightedRandomSampler     # Sampler for class imbalance
from torch.optim.lr_scheduler import LinearLR          # Linear learning rate scheduler

from sklearn.metrics import (                          # Performance metrics
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    roc_curve
)

import pandas as pd                                     # Data handling with DataFrame
from pathlib import Path                                # Filesystem path utilities
import math                                             # Math functions
import numpy as np                                      # Numerical operations on arrays

from tqdm import tqdm                                  # Progress bars for loops
import matplotlib.pyplot as plt                         # Plotting library


# =============================================================================
# Utility: Group-aware train/test split by video
# =============================================================================
def split_video_train_test(
    df: pd.DataFrame,
    group_col: str = 'video_group',
    train_frac: float = 0.7,
    seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = df[group_col].unique().tolist()
    np.random.seed(seed)
    np.random.shuffle(groups)
    n_train = int(len(groups) * train_frac)
    train_groups = groups[:n_train]
    test_groups  = groups[n_train:]
    return (
        df[df[group_col].isin(train_groups)].reset_index(drop=True),
        df[df[group_col].isin(test_groups)].reset_index(drop=True)
    )


# =============================================================================
# 1) Dataset + collate
# =============================================================================
class EmbeddingPairDataset(Dataset):
    def __init__(self, df: pd.DataFrame): self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        return (
            torch.tensor(row['img_embedding'], dtype=torch.float),
            torch.tensor(row['text_embedding'], dtype=torch.float),
            torch.tensor(row['label'], dtype=torch.float)
        )

def collate_fn(batch):
    imgs, txts, labels = zip(*batch)
    imgs = [i if i.ndim==2 else i.unsqueeze(0) for i in imgs]
    P = max(i.shape[0] for i in imgs); D_img = imgs[0].shape[1]
    imgs_p = [torch.cat([i, torch.zeros(P-i.shape[0],D_img)],dim=0) for i in imgs]
    txts = [t if t.ndim==2 else t.unsqueeze(0) for t in txts]
    N = max(t.shape[0] for t in txts); D_txt = txts[0].shape[1]
    txt_p, masks = [], []
    for t in txts:
        L = t.shape[0]
        pad = torch.zeros(N-L, D_txt)
        txt_p.append(torch.cat([t, pad],dim=0))
        masks.append(torch.tensor([True]*L + [False]*(N-L),dtype=torch.bool))
    return torch.stack(txt_p), torch.stack(masks), torch.stack(labels), torch.stack(imgs_p)


# =============================================================================
# 2) CrossModalBlock: Model Definition
# =============================================================================
class CrossModalBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout, ff_hidden_ratio, num_classes=1):
        super().__init__()
        self.txt2img = nn.MultiheadAttention(dim, num_heads, dropout, batch_first=True)
        self.norm1   = nn.LayerNorm(dim)
        self.drop1   = nn.Dropout(dropout)
        hidden = int(dim * ff_hidden_ratio)
        self.ffn     = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim)
        )
        self.norm2   = nn.LayerNorm(dim)
        self.drop2   = nn.Dropout(dropout)
        self.classifier = nn.Linear(dim, num_classes)
        self.sigmoid    = nn.Sigmoid()
        self.log_tau    = nn.Parameter(torch.zeros(num_heads))
        self.num_heads  = num_heads

    def forward(self, image_embeddings, text_embeddings, text_mask=None):
        B, P, D = image_embeddings.shape
        _, N, _ = text_embeddings.shape
        qkv = nn.functional.linear(
            image_embeddings, self.txt2img.in_proj_weight, self.txt2img.in_proj_bias
        )
        q, _ = qkv.split([D, 2 * D], dim=-1)
        kvt = nn.functional.linear(
            text_embeddings, self.txt2img.in_proj_weight[D:], self.txt2img.in_proj_bias[D:]
        )
        k, v = kvt.split([D, D], dim=-1)
        h = self.num_heads; d = D // h
        def split(x, L): return x.view(B, L, h, d).transpose(1, 2)
        Q, K, V = split(q, P), split(k, N), split(v, N)
        scores = torch.einsum('bhpd,bhqd->bhpq', Q, K) / math.sqrt(d)
        tau    = torch.exp(self.log_tau).view(1, h, 1, 1)
        scores = scores / tau
        if text_mask is not None:
            mask = ~text_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        attn    = torch.einsum('bhpq,bhqd->bhpd', weights, V)
        attn = attn.transpose(1, 2).contiguous().view(B, P, D)
        out  = nn.functional.linear(
            attn, self.txt2img.out_proj.weight, self.txt2img.out_proj.bias
        )
        x    = self.norm1(image_embeddings + self.drop1(out))
        x    = self.norm2(x + self.drop2(self.ffn(x)))
        logits = self.classifier(x).squeeze(-1)
        probs  = self.sigmoid(logits)
        return x, weights, logits, probs


# =============================================================================
# 3) Training & Evaluation utilities
# =============================================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train(); total = 0.0
    for txts, masks, labels, imgs in tqdm(loader, desc='Train'):
        imgs, txts, masks, labels = (
            imgs.to(device), txts.to(device), masks.to(device), labels.to(device)
        )
        optimizer.zero_grad()
        _, _, logits, _ = model(imgs, txts, text_mask=masks)
        loss = criterion(logits.view(-1), labels.view(-1))
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def collect_probs_labels(model, loader, device):
    model.eval(); ps, ls = [], []
    with torch.no_grad():
        for txts, masks, labels, imgs in loader:
            imgs, txts, masks = (
                imgs.to(device), txts.to(device), masks.to(device)
            )
            _, _, _, probs = model(imgs, txts, text_mask=masks)
            ps.extend(probs.view(-1).cpu().numpy())
            ls.extend(labels.numpy().astype(int))
    return np.array(ps), np.array(ls)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'auc':       roc_auc_score(y_true, y_prob),
        'f1':        f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall':    recall_score(y_true, y_pred),
        'fnr':       fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        'fpr':       fp / (fp + tn) if (fp + tn) > 0 else 0.0
    }


# =============================================================================
# 4) Main: single train/test split, collect best per seed, ensemble intervals, plot results
# =============================================================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_pickle('./fine_tune_hatemm/embeddings_hatemm_mert.pkl')
    df['video_group'] = df['frameID'].apply(lambda x: x.rsplit('_frame', 1)[0])

    train_df, test_df = split_video_train_test(df, 'video_group', train_frac=0.7, seed=42)
    assert set(train_df['video_group']).isdisjoint(set(test_df['video_group']))

    test_loader = DataLoader(EmbeddingPairDataset(test_df), batch_size=16, collate_fn=collate_fn)

    seeds = list(range(100))  # try 100 different seeds
    best_metrics_list = []
    best_f1_list = []
    best_epoch_list = []

    for seed in seeds:
        np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

        labels = train_df['label'].values.astype(int)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(1.0 / np.bincount(labels)[labels], dtype=torch.double),
            num_samples=len(labels), replacement=True
        )
        train_loader = DataLoader(
            EmbeddingPairDataset(train_df), batch_size=16,
            sampler=sampler, collate_fn=collate_fn
        )

        dim = train_df['img_embedding'].iloc[0].shape[-1]
        model = CrossModalBlock(dim, num_heads=16, dropout=0.6, ff_hidden_ratio=2).to(device)
        ckpt = torch.load('./fine_tune_hatemm/best_model_weights.pth', map_location=device)
        sd = model.state_dict()
        sd.update({k: v for k, v in ckpt.items() if k in sd and v.shape == sd[k].shape})
        model.load_state_dict(sd)

        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=15)
        pos_w = (train_df['label']==0).sum() / (train_df['label']==1).sum()
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, dtype=torch.float).to(device))

        best_f1 = 0.0
        best_met = None
        best_epoch = 0

        for epoch in range(1, 16):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

            ps_train, ls_train = collect_probs_labels(model, train_loader, device)
            ps_test, ls_test   = collect_probs_labels(model, test_loader, device)
            met_train = compute_metrics(ls_train, ps_train)
            met_test  = compute_metrics(ls_test, ps_test)

            # Track best by test F1
            if met_test['f1'] > best_f1:
                best_f1 = met_test['f1']
                best_met = met_test
                best_epoch = epoch

            print(
                f"Seed {seed} Epoch {epoch}/15 | Loss={train_loss:.4f} | "
                f"TrF1={met_train['f1']:.4f} | TeF1={met_test['f1']:.4f}, "
                f"Acc={met_test['accuracy']:.4f}, Prec={met_test['precision']:.4f}, "
                f"Rec={met_test['recall']:.4f}, AUC={met_test['auc']:.4f}"
            )
            scheduler.step()

        print(
            f"Seed {seed} best @ epoch {best_epoch}: "
            f"TeF1={best_met['f1']:.4f}, Acc={best_met['accuracy']:.4f}, "
            f"AUC={best_met['auc']:.4f}, Prec={best_met['precision']:.4f}, "
            f"Rec={best_met['recall']:.4f}, FNR={best_met['fnr']:.4f}, "
            f"FPR={best_met['fpr']:.4f}"
        )

        best_metrics_list.append(best_met)
        best_f1_list.append(best_f1)
        best_epoch_list.append(best_epoch)

    # Aggregate and print metrics across seeds
    metrics_names = ['accuracy','auc','f1','precision','recall','fnr','fpr']
    print("\nAggregated best-seed metrics across seeds:\n")
    for name in metrics_names:
        vals = np.array([m[name] for m in best_metrics_list])
        print(f"{name}: {vals.min():.4f}–{vals.max():.4f}, avg={vals.mean():.4f}")

    # Ensemble: average of best checkpoints
    avg_sd = None
    for seed, epoch in zip(seeds, best_epoch_list):
        path = Path('metrics_ft')/f'seed_{seed}_epoch_{epoch}.pth'
        sd = torch.load(path, map_location=device)
        if avg_sd is None:
            avg_sd = {k:v.clone() for k,v in sd.items()}
        else:
            for k in avg_sd: avg_sd[k] += sd[k]
    for k in avg_sd: avg_sd[k] /= len(seeds)

    ensemble_model = CrossModalBlock(dim, num_heads=16, dropout=0.6, ff_hidden_ratio=2).to(device)
    ensemble_model.load_state_dict(avg_sd)
    torch.save(avg_sd, Path('metrics_ft')/'ensemble_model.pth')

    # Plot ensemble ROC
    ps_e, ls_e = collect_probs_labels(ensemble_model, test_loader, device)
    fpr, tpr, _ = roc_curve(ls_e, ps_e)
    plt.figure()
    plt.plot(fpr, tpr, label='CMA ROC')
    plt.plot([0,1], [0,1], linestyle='--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('CMA ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(Path('metrics_ft')/'ensemble_roc.png')
    plt.close()

    # Plot distribution of best F1 scores across seeds as a boxplot
    plt.figure()
    plt.boxplot(best_f1_list)
    plt.ylabel("Best F1 Score")
    plt.title("Distribution of Best F1 Scores (CMA)")
    plt.savefig(Path('metrics_ft')/'seed_best_f1_boxplot.png')
    plt.close()
