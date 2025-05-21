import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import MultiheadAttention, LayerNorm, Sigmoid
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
import pandas as pd
from pathlib import Path
import math
import numpy as np
from tqdm import tqdm

# ============================================================================ 
# Device setup
# ============================================================================ 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[Device] Using device: {device}')

# ============================================================================ 
# Dataset class & collate_fn
# ============================================================================ 
class EmbeddingPairDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.reset_index(drop=True)
        print(f'[Dataset] Initialized with {len(self.df)} samples')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        return (
            torch.tensor(row['img_embedding'], dtype=torch.float),
            torch.tensor(row['text_embedding'], dtype=torch.float),
            torch.tensor(row['label'], dtype=torch.float),
        )

def collate_fn(batch):
    imgs, txts, labels = zip(*batch)

    # --- Image padding
    imgs = [i if i.ndim == 2 else i.unsqueeze(0) for i in imgs]
    max_p = max(i.shape[0] for i in imgs)
    d_img = imgs[0].shape[1]
    padded_imgs = []
    for i in imgs:
        if i.shape[0] < max_p:
            pad = torch.zeros(max_p - i.shape[0], d_img, dtype=i.dtype)
            padded_imgs.append(torch.cat([i, pad], dim=0))
        else:
            padded_imgs.append(i)
    imgs_tensor = torch.stack(padded_imgs)  # (B, P, D)

    # --- Text padding & mask
    txts = [t if t.ndim == 2 else t.unsqueeze(0) for t in txts]
    max_n = max(t.shape[0] for t in txts)
    d_txt = txts[0].shape[1]
    padded_txts, masks = [], []
    for t in txts:
        n = t.shape[0]
        if n < max_n:
            pad = torch.zeros(max_n - n, d_txt, dtype=t.dtype)
            t = torch.cat([t, pad], dim=0)
        padded_txts.append(t)
        masks.append(torch.tensor([True]*n + [False]*(max_n-n), dtype=torch.bool))

    txts_tensor = torch.stack(padded_txts)   # (B, N, D_txt)
    masks_tensor = torch.stack(masks)        # (B, N)
    labels_tensor = torch.stack(labels)      # (B, ...)

    return imgs_tensor, txts_tensor, masks_tensor, labels_tensor

# ============================================================================ 
# Model definition
# ============================================================================ 
class CrossModalBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout, ff_hidden_ratio=2, num_classes=1):
        super().__init__()
        self.txt2img = MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = LayerNorm(dim)
        self.drop1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_hidden_ratio),
            nn.ReLU(),
            nn.Linear(dim * ff_hidden_ratio, dim),
        )
        self.norm2 = LayerNorm(dim)
        self.drop2 = nn.Dropout(dropout)

        self.classifier = nn.Linear(dim, num_classes)
        self.sigmoid = Sigmoid()
        self.log_tau = nn.Parameter(torch.zeros(num_heads))
        self.num_heads = num_heads

    def forward(self, image_embeddings, text_embeddings, text_mask=None):
        B, P, D = image_embeddings.shape

        # project Q from image
        qkv = nn.functional.linear(
            image_embeddings,
            self.txt2img.in_proj_weight,
            self.txt2img.in_proj_bias
        )
        q, _ = qkv.split([D, 2*D], dim=-1)

        # project K,V from text
        kvt = nn.functional.linear(
            text_embeddings,
            self.txt2img.in_proj_weight[D:],
            self.txt2img.in_proj_bias[D:]
        )
        k, v = kvt.split([D, D], dim=-1)

        # reshape for multi-head
        h = self.num_heads
        d_head = D // h
        Q = q.view(B, P, h, d_head).transpose(1, 2)   # (B,h,P,d)
        K = k.view(B, -1, h, d_head).transpose(1, 2)  # (B,h,N,d)
        V = v.view(B, -1, h, d_head).transpose(1, 2)

        # scaled dot-product + learnable temperature
        scores = torch.einsum('bhpd,bhqd->bhpq', Q, K) / math.sqrt(d_head)
        tau = torch.exp(self.log_tau).view(1, h, 1, 1)
        scores = scores / tau

        if text_mask is not None:
            mask = ~text_mask.unsqueeze(1).unsqueeze(2)  # broadcast to (B,h,1,N)
            scores = scores.masked_fill(mask, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        attn = torch.einsum('bhpq,bhqd->bhpd', weights, V)
        attn = attn.transpose(1, 2).reshape(B, P, D)

        img_up = nn.functional.linear(
            attn,
            self.txt2img.out_proj.weight,
            self.txt2img.out_proj.bias
        )

        x = self.norm1(image_embeddings + self.drop1(img_up))
        x = self.norm2(x + self.drop2(self.ffn(x)))

        logits = self.classifier(x).squeeze(-1)  # (B, P)
        return x, weights, logits, self.sigmoid(logits)

# ============================================================================ 
# Training utilities
# ============================================================================ 
def split_datasets(df, test_size=10000, val_size=5000, seed=42):
    test_df = df.sample(n=test_size, random_state=seed)
    rem = df.drop(test_df.index)
    val_df = rem.sample(n=val_size, random_state=seed)
    train_df = rem.drop(val_df.index)
    print(f'[Data] train={len(train_df)}, val={len(val_df)}, test={len(test_df)}')
    return train_df, val_df, test_df

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for imgs, txts, mask, labels in tqdm(loader, desc='Training'):
        imgs = imgs.to(device)
        txts = txts.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        _, _, logits, _ = model(imgs, txts, text_mask=mask)
        loss = criterion(logits.view(-1), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg = total_loss / len(loader)
    print(f'[Train] Avg loss: {avg:.4f}')
    return avg

def evaluate(model, loader, stage='Val'):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, txts, mask, labels in tqdm(loader, desc=f'{stage} eval'):
        imgs = imgs.to(device)
        txts = txts.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            _, _, _, probs = model(imgs, txts, text_mask=mask)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    probs = np.concatenate(all_probs).ravel()
    labels = np.concatenate(all_labels).ravel()
    preds = (probs > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds),
        'roc_auc': roc_auc_score(labels, probs),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'FNR': fn / (fn + tp),
        'FPR': fp / (fp + tn),
    }
    print(f'[{stage}] ' + ', '.join(f'{k}: {v:.4f}' for k, v in metrics.items()))
    return metrics

# ============================================================================ 
# Main training loop
# ============================================================================ 
def run_training(df, hp_grid):
    train_df, val_df, test_df = split_datasets(df)
    dim = train_df['img_embedding'].iloc[0].shape[-1]
    best_model, best_metrics = None, {}

    print("Training started....")

    results = []
    for lr in hp_grid['lr']:
        for wd in hp_grid['weight_decay']:
            for heads in hp_grid['num_heads']:
                for drop in hp_grid['dropout']:
                    print(f'\n[Config] lr={lr}, wd={wd}, heads={heads}, drop={drop}')

                    train_loader = DataLoader(
                        EmbeddingPairDataset(train_df),
                        batch_size=12,
                        shuffle=True,
                        collate_fn=collate_fn
                    )
                    val_loader = DataLoader(
                        EmbeddingPairDataset(val_df),
                        batch_size=12,
                        collate_fn=collate_fn
                    )

                    model = CrossModalBlock(dim, heads, drop).to(device)
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

                    # handle class imbalance
                    pos_w = (len(train_df) - train_df.label.sum()) / train_df.label.sum()
                    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))

                    for epoch in range(1, hp_grid['epochs'] + 1):
                        print(f'Epoch {epoch}/{hp_grid["epochs"]}')
                        train_one_epoch(model, train_loader, optimizer, criterion)
                        val_metrics = evaluate(model, val_loader, stage='Val')

                        if not best_metrics or val_metrics['FNR'] < best_metrics['FNR']:
                            best_metrics = val_metrics
                            best_model = model
                            torch.save(model.state_dict(), 'best_model.pth')

                    results.append({
                        'lr': lr,
                        'weight_decay': wd,
                        'heads': heads,
                        'dropout': drop,
                        **val_metrics
                    })

    # final test
    test_loader = DataLoader(
        EmbeddingPairDataset(test_df),
        batch_size=12,
        collate_fn=collate_fn
    )
    test_metrics = evaluate(best_model, test_loader, stage='Test')

    # save CSVs
    pd.DataFrame(results).to_csv('validation_results.csv', index=False)
    pd.DataFrame([test_metrics]).to_csv('test_results.csv', index=False)

    return best_model

if __name__ == '__main__':
    df = pd.read_pickle(Path.cwd() / 'results' / 'fusion_data.pkl')
    #df = pd.read_pickle(Path('fusion_data.pkl'))
    print("Data is loaded...")
    hp_grid = {
        'lr': [1e-3, 1e-6],
        'weight_decay': [0.01, 0.1],
        'num_heads': [4,12],
        'dropout': [0.2, 0.6],
        'epochs': 3,
    }
    best_model = run_training(df, hp_grid)
    print('Training complete. Results saved to validation_results.csv and test_results.csv')




