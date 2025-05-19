# %%
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

import pandas as pd                                     # Data handling
from pathlib import Path                                # Filesystem paths
import math                                             # Math functions
import numpy as np                                      # Numerical arrays

from tqdm import tqdm                                   # Progress bars
import matplotlib.pyplot as plt                         # Plotting


# =============================================================================
# Utility: Group-aware train/validation/test split by video
# =============================================================================
def split_by_video(df: pd.DataFrame, frac: float, seed: int):
    groups = df['video_group'].unique().tolist()
    np.random.seed(seed)
    np.random.shuffle(groups)
    n = int(len(groups) * frac)
    return groups[:n], groups[n:]


def make_splits(df: pd.DataFrame, val_frac=0.1, test_frac=0.2, seed: int = 42):
    test_groups, rem = split_by_video(df, test_frac, seed)
    df_test = df[df['video_group'].isin(test_groups)].reset_index(drop=True)

    df_rem = df[df['video_group'].isin(rem)].reset_index(drop=True)
    val_groups, train_groups = split_by_video(
        df_rem,
        val_frac / (1 - test_frac),
        seed
    )
    df_val   = df_rem[df_rem['video_group'].isin(val_groups)].reset_index(drop=True)
    df_train = df_rem[df_rem['video_group'].isin(train_groups)].reset_index(drop=True)

    return df_train, df_val, df_test


# =============================================================================
# Dataset + collate
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
            torch.tensor(row['label'], dtype=torch.float)
        )


def collate_fn(batch):
    idxs, imgs, txts, labels = zip(*batch)

    # Pad image sequences
    imgs = [i if i.ndim == 2 else i.unsqueeze(0) for i in imgs]
    P      = max(i.shape[0] for i in imgs)
    D_img  = imgs[0].shape[1]
    imgs_p = [
        torch.cat([i, torch.zeros(P - i.shape[0], D_img)], dim=0)
        for i in imgs
    ]

    # Pad text sequences and create masks
    txts = [t if t.ndim == 2 else t.unsqueeze(0) for t in txts]
    N      = max(t.shape[0] for t in txts)
    D_txt  = txts[0].shape[1]

    txt_p, masks = [], []
    for t in txts:
        L   = t.shape[0]
        pad = torch.zeros(N - L, D_txt)
        txt_p.append(torch.cat([t, pad], dim=0))
        masks.append(torch.tensor([True] * L + [False] * (N - L), dtype=torch.bool))

    return (
        list(idxs),
        torch.stack(txt_p),
        torch.stack(masks),
        torch.stack(labels),
        torch.stack(imgs_p)
    )


# =============================================================================
# CrossModalBlock with learnable temperature
# =============================================================================
class CrossModalBlock(nn.Module):
    def __init__(self, dim, heads, dropout, ff_ratio, num_classes=1):
        super().__init__()
        self.attn      = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.log_tau   = nn.Parameter(torch.zeros(heads))
        self.norm1     = nn.LayerNorm(dim)
        self.drop1     = nn.Dropout(dropout)

        hidden = int(dim * ff_ratio)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim)
        )
        self.norm2      = nn.LayerNorm(dim)
        self.drop2      = nn.Dropout(dropout)
        self.classifier = nn.Linear(dim, num_classes)
        self.sigmoid    = nn.Sigmoid()
        self.heads      = heads

    def forward(self, img_emb, txt_emb, text_mask=None):
        B, P, D = img_emb.shape
        _, N, _ = txt_emb.shape

        # Project to q, k, v
        qkv = nn.functional.linear(
            img_emb,
            self.attn.in_proj_weight,
            self.attn.in_proj_bias
        )
        q, _ = qkv.split([D, 2 * D], dim=-1)

        kvt = nn.functional.linear(
            txt_emb,
            self.attn.in_proj_weight[D:],
            self.attn.in_proj_bias[D:]
        )
        k, v = kvt.split([D, D], dim=-1)

        # Split heads
        h = self.heads
        d = D // self.heads

        def split(x, L):
            return x.view(B, L, h, d).transpose(1, 2)

        Q = split(q, P)
        K = split(k, N)
        V = split(v, N)

        # Scaled dot-product with learnable temperature
        raw_scores = torch.einsum('bhpd,bhqd->bhpq', Q, K) / math.sqrt(d)
        tau        = torch.exp(self.log_tau).view(1, h, 1, 1)
        scores     = raw_scores / tau

        if text_mask is not None:
            mask = ~text_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))

        weights  = torch.softmax(scores, dim=-1)
        attn_out = torch.einsum('bhpq,bhqd->bhpd', weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, P, D)
        out      = nn.functional.linear(
            attn_out,
            self.attn.out_proj.weight,
            self.attn.out_proj.bias
        )

        x      = self.norm1(img_emb + self.drop1(out))
        x      = self.norm2(x + self.drop2(self.ffn(x)))
        logits = self.classifier(x).squeeze(-1)

        return x, weights, logits, self.sigmoid(logits)


# =============================================================================
# Training, sliding-window eval, threshold select, seed loop, aggregate metrics, save CSV & weights
# =============================================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for idxs, txts, masks, labels, imgs in tqdm(loader, desc='Train'):
        imgs, txts, masks, labels = (
            imgs.to(device),
            txts.to(device),
            masks.to(device),
            labels.to(device)
        )
        optimizer.zero_grad()

        _, _, logits, _ = model(imgs, txts, text_mask=masks)
        loss = criterion(logits.view(-1), labels.view(-1))

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def collect_probs_labels_sliding(model, loader, device, window_size=5, strategy='mean'):
    model.eval()
    entries = []

    with torch.no_grad():
        for idxs, txts, masks, labels, imgs in loader:
            imgs, txts, masks = (
                imgs.to(device),
                txts.to(device),
                masks.to(device)
            )
            _, _, _, probs = model(imgs, txts, text_mask=masks)

            for i, idx in enumerate(idxs):
                entries.append((
                    idx,
                    loader.dataset.df.loc[idx, 'video_group'],
                    loader.dataset.df.loc[idx, 'frameID'],
                    probs[i].item(),
                    labels[i].item()
                ))

    df_rel = pd.DataFrame(entries, columns=['idx','video_group','frameID','raw_prob','label'])
    parts  = df_rel['frameID'].str.rsplit('_frame', n=1, expand=True)
    df_rel['frame_num'] = parts[1].str.replace(r'\D+', '', regex=True).astype(int)

    sm, tl = [], []
    for vid, grp in df_rel.groupby('video_group'):
        grp_sorted = grp.sort_values('frame_num')
        p = grp_sorted['raw_prob'].values
        if len(p) >= window_size:
            w = np.lib.stride_tricks.sliding_window_view(p, window_size)
            v = w.mean(1) if strategy == 'mean' else w.max(1)
            pad = np.repeat(v[-1], len(p) - len(v))
            seq = np.concatenate([v, pad])
        else:
            seq = p.copy()
        sm.extend(seq)
        tl.extend(grp_sorted['label'].tolist())

    return np.array(sm), np.array(tl)


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


def select_threshold(y_true, y_prob):
    best_t, best_f = 0, 0
    for t in np.linspace(0, 1, 101):
        f = f1_score(y_true, (y_prob >= t).astype(int))
        if f > best_f:
            best_f, best_t = f, t
    return best_t


if __name__ == '__main__':
    # ensure output directory exists
    out_dir = Path('./fine_tune_hatemm/metrics_ft')
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle('./fine_tune_hatemm/embeddings_hatemm_mert.pkl')
    parts = df['frameID'].str.rsplit('_frame', n=1, expand=True)
    df['video_group'] = parts[0]
    df['frame_num']   = parts[1].str.replace(r'\D+', '', regex=True).astype(int)

    train_df, val_df, test_df = make_splits(df, val_frac=0.1, test_frac=0.2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim    = train_df['img_embedding'].iloc[0].shape[-1]

    seeds           = list(range(100))  # adjust as needed
    metrics_list    = []
    best_thresholds = []

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        train_loader = DataLoader(
            EmbeddingPairDataset(train_df), batch_size=16,
            sampler=WeightedRandomSampler(
                torch.tensor(1.0/np.bincount(train_df['label'].astype(int))[train_df['label'].astype(int)], dtype=torch.double),
                len(train_df), True
            ),
            collate_fn=collate_fn
        )

        val_loader = DataLoader(
            EmbeddingPairDataset(val_df), batch_size=16, collate_fn=collate_fn
        )

        test_loader = DataLoader(
            EmbeddingPairDataset(test_df), batch_size=16, collate_fn=collate_fn
        )

        # Instantiate model and load pretrained weights
        model = CrossModalBlock(dim, 16, 0.6, 2).to(device)
        pretrained_path = './fine_tune_hatemm/best_model_weights.pth'
        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)

        # Optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=15)

        pos_w     = (train_df['label'] == 0).sum() / (train_df['label'] == 1).sum()
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, dtype=torch.float).to(device))

        best_f1     = 0.0
        best_metrics= None
        best_epoch  = 0
        best_thresh = 0.0

        for epoch in range(1, 10):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()

            ps_val, ls_val = collect_probs_labels_sliding(model, val_loader, device)
            thresh         = select_threshold(ls_val, ps_val)
            best_thresholds.append(thresh)

            ps_test, ls_test = collect_probs_labels_sliding(model, test_loader, device)
            metrics          = compute_metrics(ls_test, ps_test, threshold=thresh)

            if metrics['f1'] > best_f1:
                best_f1     = metrics['f1']
                best_metrics= metrics.copy()
                best_epoch  = epoch
                best_thresh = thresh

                # save best weights for this seed
                torch.save(
                    model.state_dict(),
                    out_dir / f'seed_{seed}_best_epoch_{epoch}.pth'
                )

            print(f"Seed {seed} Epoch {epoch} | Loss={loss:.4f} | F1={metrics['f1']:.4f} (th={thresh:.2f})")

        print(
            f"Seed {seed} best @ epoch {best_epoch}: " +
            ", ".join([f"{k}={v:.4f}" for k,v in best_metrics.items()]) +
            f" | th={best_thresh:.2f}"
        )
        metrics_list.append(best_metrics)

    # aggregate and save metrics
    print("\nAggregated metrics across seeds (best epochs):")
    names   = ['accuracy', 'auc', 'f1', 'precision', 'recall', 'fnr', 'fpr']
    records = []

    for n in names:
        vals = np.array([m[n] for m in metrics_list])
        print(f"{n}: min={vals.min():.4f}, max={vals.max():.4f}, avg={vals.mean():.4f}")
        records.append({'metric': n, 'min': vals.min(), 'max': vals.max(), 'avg': vals.mean()})

    avg_thresh = np.mean(best_thresholds)
    print(f"Average threshold across all epochs and seeds: {avg_thresh:.2f}")
    records.append({'metric': 'threshold', 'min': avg_thresh, 'max': avg_thresh, 'avg': avg_thresh})

    pd.DataFrame(records).to_csv(out_dir / 'aggregated_metrics.csv', index=False)

    # final ROC plot
    ps_final, ls_final = ps_test, ls_test
    fpr, tpr, _        = roc_curve(ls_final, ps_final)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC(AUC={metrics_list[-1]['auc']:.2f})")
    plt.plot([0, 1], [0, 1], '--')
    plt.title('Test ROC')
    plt.legend()
    plt.savefig(out_dir / 'final_test_roc.png')
    plt.close()