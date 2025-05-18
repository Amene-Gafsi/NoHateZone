import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import MultiheadAttention, LayerNorm, Sigmoid
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import pandas as pd
from pathlib import Path
import math
import numpy as np
from tqdm import tqdm  # for progress bars
import matplotlib.pyplot as plt

# ============================================================================
# Device setup
# ============================================================================
# Select GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[Device] Using device: {device}')

# ============================================================================
# Dataset class & collate_fn
# ============================================================================
# PyTorch Dataset for paired image and text embeddings with labels
class EmbeddingPairDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        # Reset index for safe indexing and store DataFrame
        print(f'[Dataset] Initializing with {len(dataframe)} samples')
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        # Return number of samples
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve image, text embeddings and label for a single sample
        row = self.df.loc[idx]
        img = torch.tensor(row['img_embedding'], dtype=torch.float)
        txt = torch.tensor(row['text_embedding'], dtype=torch.float)
        lbl = torch.tensor(row['label'], dtype=torch.float)
        return img, txt, lbl

# Custom collate function to pad variable-length sequences
def collate_fn(batch):
    imgs, txts, labels = zip(*batch)
    # Pad image embeddings to max sequence length in batch
    imgs = [img if img.ndim == 2 else img.unsqueeze(0) for img in imgs]
    max_p = max(i.shape[0] for i in imgs)
    d_img = imgs[0].shape[1]
    padded_imgs = [
        torch.cat([i, torch.zeros(max_p - i.shape[0], d_img)], 0)
        if i.shape[0] < max_p else i
        for i in imgs
    ]
    imgs_tensor = torch.stack(padded_imgs)

    # Pad text embeddings and create mask
    txts = [t if t.ndim == 2 else t.unsqueeze(0) for t in txts]
    max_n = max(t.shape[0] for t in txts)
    d_txt = txts[0].shape[1]
    padded_txts, masks = [], []
    for t in txts:
        n = t.shape[0]
        if n < max_n:
            t = torch.cat([t, torch.zeros(max_n - n, d_txt)], 0)
        mask = torch.tensor([True] * n + [False] * (max_n - n), dtype=torch.bool)
        padded_txts.append(t)
        masks.append(mask)

    txts_tensor = torch.stack(padded_txts)
    mask_tensor = torch.stack(masks)
    labels_tensor = torch.stack(labels)
    return imgs_tensor, txts_tensor, mask_tensor, labels_tensor

# ============================================================================
# Model definition
# ============================================================================
# Cross-modal attention block: attends from image to text embeddings
class CrossModalBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout, ff_hidden_ratio, num_classes=1):
        super().__init__()
        # Multi-head attention from image queries to text keys/values
        self.txt2img = MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = LayerNorm(dim)
        self.drop1 = nn.Dropout(dropout)

        # Feed-forward network
        hidden = int(dim * ff_hidden_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim)
        )
        self.norm2 = LayerNorm(dim)
        self.drop2 = nn.Dropout(dropout)

        # Classification head
        self.classifier = nn.Linear(dim, num_classes)
        self.sigmoid = Sigmoid()

        # Learnable temperature for scaling attention scores
        self.log_tau = nn.Parameter(torch.zeros(num_heads))
        self.num_heads = num_heads

    def forward(self, image_embeddings, text_embeddings, text_mask=None):
        # image_embeddings: (B, P, D), text_embeddings: (B, N, D)
        B, P, D = image_embeddings.shape
        _, n, D2 = text_embeddings.shape
        assert D == D2, 'Embedding dims must match'

        # Project to queries, keys, values
        qkv = nn.functional.linear(
            image_embeddings,
            self.txt2img.in_proj_weight,
            self.txt2img.in_proj_bias
        )
        q, _ = qkv.split([D, 2 * D], -1)
        kvt = nn.functional.linear(
            text_embeddings,
            self.txt2img.in_proj_weight[D:],
            self.txt2img.in_proj_bias[D:]
        )
        k, v = kvt.split([D, D], -1)

        # Split into multiple heads
        h = self.num_heads
        d_head = D // h
        def split_heads(x, L):
            return x.view(B, L, h, d_head).transpose(1, 2)

        Q = split_heads(q, P)
        K = split_heads(k, n)
        V = split_heads(v, n)

        # Compute scaled dot-product attention
        scores = torch.einsum('bhpd,bhqd->bhpq', Q, K) / math.sqrt(d_head)
        tau = torch.exp(self.log_tau).view(1, h, 1, 1)
        scores = scores / tau

        # Apply mask if provided
        if text_mask is not None:
            mask = ~text_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))

        weights = torch.softmax(scores, -1)
        attn = torch.einsum('bhpq,bhqd->bhpd', weights, V)
        attn = attn.transpose(1, 2).contiguous().view(B, P, D)

        # Output projection
        img_up = nn.functional.linear(
            attn,
            self.txt2img.out_proj.weight,
            self.txt2img.out_proj.bias
        )

        # Add & Norm, then Feed-forward & Norm
        x = self.norm1(image_embeddings + self.drop1(img_up))
        x = self.norm2(x + self.drop2(self.ffn(x)))

        # Classification logits and probabilities
        logits = self.classifier(x).squeeze(-1)
        probs = self.sigmoid(logits)

        return x, weights, logits, probs

# ============================================================================
# Training and evaluation utilities
# ============================================================================
def split_datasets(df, seed=42):
    # Split DataFrame: train 60%, val 20%, test 20%
    train, temp = train_test_split(df, test_size=0.4, random_state=seed)
    val, test = train_test_split(temp, test_size=0.5, random_state=seed)
    print(f'[Data] train={len(train)}, val={len(val)}, test={len(test)}')
    return train, val, test


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for imgs, txts, mask, labels in tqdm(loader, desc='Train batches'):
        imgs, txts, mask, labels = imgs.to(device), txts.to(device), mask.to(device), labels.to(device)
        optimizer.zero_grad()
        _, _, logits, _ = model(imgs, txts, text_mask=mask)
        loss = criterion(logits.view(-1), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg = total_loss / len(loader)
    print(f'[Train] Avg loss: {avg:.4f}')
    return avg


def eval_metrics(model, loader, stage='Eval'):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, txts, mask, labels in tqdm(loader, desc=f'{stage} batches'):
        imgs, txts, mask = imgs.to(device), txts.to(device), mask.to(device)
        with torch.no_grad():
            _, _, _, probs = model(imgs, txts, text_mask=mask)
        all_probs.extend(probs.view(-1).cpu().numpy())
        all_labels.extend(labels.numpy().astype(int))
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    stats = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'FNR': fn / (fn + tp),
        'FPR': fp / (fp + tn)
    }
    print(f'[{stage}] ' + ', '.join([f'{k}: {v:.4f}' for k, v in stats.items()]))
    return stats


def run_training(df, hp_grid):
    train_df, val_df, test_df = split_datasets(df)
    dim = train_df['img_embedding'].iloc[0].shape[-1]
    best_val_fnr, best_cfg, best_losses = float('inf'), None, None
    results = []
    for lr in hp_grid['lr']:
        for bs in hp_grid['batch_size']:
            for heads in hp_grid['num_heads']:
                for drop in hp_grid['dropout']:
                    for ff_r in hp_grid['ff_hidden_ratio']:
                        print(f'[Config] lr={lr}, bs={bs}, heads={heads}, drop={drop}, ff_r={ff_r}')
                        tr_loader = DataLoader(EmbeddingPairDataset(train_df), batch_size=bs, collate_fn=collate_fn, shuffle=True)
                        va_loader = DataLoader(EmbeddingPairDataset(val_df), batch_size=bs, collate_fn=collate_fn)
                        model = CrossModalBlock(dim, heads, drop, ff_r).to(device)
                        optimizer = optim.Adam(model.parameters(), lr=lr)
                        wt = torch.tensor(train_df['label'].values, dtype=torch.float)
                        pos_weight = (wt == 0).sum() / (wt == 1).sum()
                        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
                        losses = []
                        for e in range(1, hp_grid['epochs'] + 1):
                            print('-- Epoch {}/{}'.format(e, hp_grid['epochs']))
                            loss = train_one_epoch(model, tr_loader, optimizer, criterion)
                            losses.append(loss)
                            _ = eval_metrics(model, va_loader, stage='Val')
                        final_metrics = eval_metrics(model, va_loader, stage='Val final')
                        results.append({**{'lr': lr, 'batch_size': bs, 'num_heads': heads, 'dropout': drop, 'ff_hidden_ratio': ff_r}, **final_metrics})
                        if final_metrics['FNR'] < best_val_fnr:
                            best_val_fnr = final_metrics['FNR']
                            best_model, best_cfg, best_losses = model, (lr, bs, heads, drop, ff_r), losses
    Path('metrics').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(Path('metrics') / 'validation_metrics.csv', index=False)
    print(f'[Test] Best config: {best_cfg}')
    test_loader = DataLoader(EmbeddingPairDataset(test_df), batch_size=best_cfg[1], collate_fn=collate_fn)
    test_stats = eval_metrics(best_model, test_loader, stage='Test')
    pd.DataFrame([test_stats]).to_csv(Path('metrics') / 'best_model_test_metrics.csv', index=False)
    torch.save(best_model.state_dict(), Path('models') / 'best_model_weights.pth')
    plt.figure()
    plt.plot(range(1, len(best_losses) + 1), best_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Best Model Training Loss')
    plt.savefig(Path('metrics') / 'best_model_loss.png')
    return best_model

if __name__ == '__main__':
    df = pd.read_pickle(Path.cwd() / 'results' / 'fusion_data.pkl')
    hp_grid = {
        'lr': [1e-3, 1e-5, 1e-6],
        'batch_size': [8, 12, 16],
        'num_heads': [8, 12, 16],
        'dropout': [0.1, 0.4],
        'ff_hidden_ratio': [2, 4],
        'epochs': 3
    }
    run_training(df, hp_grid)
    print('Done. Metrics, model weights, and loss plot saved.')
