import torch  # Core PyTorch library for tensor operations and model building
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
from torch.nn import MultiheadAttention, LayerNorm, Sigmoid  # Specific layers from PyTorch
from torch.utils.data import DataLoader, Dataset  # Utilities for batching and dataset management
from sklearn.model_selection import train_test_split  # Dataset splitting
from sklearn.metrics import (  # Performance metrics
    confusion_matrix, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
)
import pandas as pd  # Data manipulation with DataFrame
from pathlib import Path  # Filesystem path utilities
import math  # Math utilities (e.g., sqrt)
import numpy as np  # Numerical operations
from tqdm import tqdm  # Progress bars for loops
import matplotlib.pyplot as plt  # Plotting library

# ============================================================================
# Device setup: Choose GPU if available, else CPU
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[Device] Using device: {device}')  # Log selected device

# ============================================================================
# Dataset class & collate_fn: Handle paired embeddings with variable lengths
# ============================================================================
class EmbeddingPairDataset(Dataset):
    """
    PyTorch Dataset returning image-text embedding pairs and binary labels.
    Expects a DataFrame with 'img_embedding', 'text_embedding', and 'label' columns.
    """
    def __init__(self, dataframe: pd.DataFrame):
        print(f'[Dataset] Initializing with {len(dataframe)} samples')
        # Reset index for safe integer-based indexing
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        # Total number of samples
        return len(self.df)

    def __getitem__(self, idx):
        # Fetch row by index
        row = self.df.loc[idx]
        img = torch.tensor(row['img_embedding'], dtype=torch.float)  # Image embedding tensor
        txt = torch.tensor(row['text_embedding'], dtype=torch.float)  # Text embedding tensor
        lbl = torch.tensor(row['label'], dtype=torch.float)  # Binary label tensor
        return img, txt, lbl  # Return tuple for collate_fn


def collate_fn(batch):
    """
    Custom collate to pad variable-length image/text sequences in a batch.
    Returns padded tensors and masks for text embeddings.
    """
    imgs, txts, labels = zip(*batch)
    # Ensure each embedding has shape [seq_len, dim]
    imgs = [img if img.ndim == 2 else img.unsqueeze(0) for img in imgs]
    # Determine max image sequence length in batch
    max_p = max(i.shape[0] for i in imgs)
    d_img = imgs[0].shape[1]
    # Pad shorter image sequences with zeros
    padded_imgs = [
        torch.cat([i, torch.zeros(max_p - i.shape[0], d_img)], dim=0) if i.shape[0] < max_p else i
        for i in imgs
    ]
    imgs_tensor = torch.stack(padded_imgs)  # [B, max_p, d_img]

    # Similar padding for text embeddings
    txts = [t if t.ndim == 2 else t.unsqueeze(0) for t in txts]
    max_n = max(t.shape[0] for t in txts)
    d_txt = txts[0].shape[1]
    padded_txts, masks = [], []
    for t in txts:
        n = t.shape[0]
        if n < max_n:
            # Pad text embedding
            t = torch.cat([t, torch.zeros(max_n - n, d_txt)], dim=0)
        # Create mask: True for valid tokens, False for padding
        mask = torch.tensor([True] * n + [False] * (max_n - n), dtype=torch.bool)
        padded_txts.append(t)
        masks.append(mask)

    txts_tensor = torch.stack(padded_txts)  # [B, max_n, d_txt]
    mask_tensor = torch.stack(masks)        # [B, max_n]
    labels_tensor = torch.stack(labels)     # [B]
    return imgs_tensor, txts_tensor, mask_tensor, labels_tensor

# ============================================================================
# Model definition: Cross-modal attention block
# ============================================================================
class CrossModalBlock(nn.Module):
    """
    Single block performing attention from image query to text keys/values,
    followed by feed-forward layers and binary classification.
    """
    def __init__(self, dim, num_heads, dropout, ff_hidden_ratio, num_classes=1):
        super().__init__()
        # Multi-head attention: image embeddings query text embeddings
        self.txt2img = MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = LayerNorm(dim)  # Layer norm after attention
        self.drop1 = nn.Dropout(dropout)

        # Feed-forward network: expand and contract feature dimension
        hidden = int(dim * ff_hidden_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim)
        )
        self.norm2 = LayerNorm(dim)  # Layer norm after FFN
        self.drop2 = nn.Dropout(dropout)

        # Final classification head projecting each image token to a logit
        self.classifier = nn.Linear(dim, num_classes)
        self.sigmoid = Sigmoid()  # Sigmoid activation for binary probability

        # Learnable temperature for attention scaling per head
        self.log_tau = nn.Parameter(torch.zeros(num_heads))
        self.num_heads = num_heads

    def forward(self, image_embeddings, text_embeddings, text_mask=None):
        """
        Args:
            image_embeddings: [B, P, D]
            text_embeddings: [B, N, D]
            text_mask: [B, N] boolean mask
        Returns:
            x: Updated image embeddings
            weights: Attention weights [B, heads, P, N]
            logits: Raw classification scores [B, P]
            probs: Sigmoid probabilities [B, P]
        """
        B, P, D = image_embeddings.shape
        _, N, D2 = text_embeddings.shape
        assert D == D2, 'Embedding dims must match'

        # Project inputs into Q, K, V using in_proj_weight/bias
        qkv = nn.functional.linear(
            image_embeddings, self.txt2img.in_proj_weight, self.txt2img.in_proj_bias
        )
        # Split qkv: q=[B,P,D], rest for K & V
        q, _ = qkv.split([D, 2 * D], dim=-1)
        kvt = nn.functional.linear(
            text_embeddings,
            self.txt2img.in_proj_weight[D:],
            self.txt2img.in_proj_bias[D:]
        )
        # Split into key and value
        k, v = kvt.split([D, D], dim=-1)

        # Reshape for multi-head: [B, heads, seq_len, head_dim]
        h = self.num_heads
        d_head = D // h
        def split_heads(x, L):
            return x.view(B, L, h, d_head).transpose(1, 2)

        Q = split_heads(q, P)
        K = split_heads(k, N)
        V = split_heads(v, N)

        # Scaled dot-product attention with learnable temperature
        scores = torch.einsum('bhpd,bhqd->bhpq', Q, K) / math.sqrt(d_head)
        tau = torch.exp(self.log_tau).view(1, h, 1, 1)
        scores = scores / tau

        # Apply text mask (pad tokens get -inf score)
        if text_mask is not None:
            mask = ~text_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        # Weighted sum to produce attended features
        attn = torch.einsum('bhpq,bhqd->bhpd', weights, V)
        # Restore shape [B, P, D]
        attn = attn.transpose(1, 2).contiguous().view(B, P, D)

        # Output projection
        img_up = nn.functional.linear(
            attn, self.txt2img.out_proj.weight, self.txt2img.out_proj.bias
        )

        # Residual, dropout, norm, then FFN + residual + norm
        x = self.norm1(image_embeddings + self.drop1(img_up))
        x = self.norm2(x + self.drop2(self.ffn(x)))

        # Classification per image token
        logits = self.classifier(x).squeeze(-1)  # [B, P]
        probs = self.sigmoid(logits)             # [B, P]

        return x, weights, logits, probs

# ============================================================================
# Training and evaluation utilities
# ============================================================================
def split_datasets(df, seed=42):
    """
    Split into train/val/test with ratios 60/20/20.
    """
    train, temp = train_test_split(df, test_size=0.4, random_state=seed)
    val, test = train_test_split(temp, test_size=0.5, random_state=seed)
    print(f'[Data] train={len(train)}, val={len(val)}, test={len(test)}')
    return train, val, test


def train_one_epoch(model, loader, optimizer, criterion):
    """
    Run one epoch of training.
    Returns average loss.
    """
    model.train()
    total_loss = 0
    for imgs, txts, mask, labels in tqdm(loader, desc='Train batches'):
        imgs, txts, mask, labels = imgs.to(device), txts.to(device), mask.to(device), labels.to(device)
        optimizer.zero_grad()
        _, _, logits, _ = model(imgs, txts, text_mask=mask)
        # BCEWithLogitsLoss expects raw logits
        loss = criterion(logits.view(-1), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg = total_loss / len(loader)
    print(f'[Train] Avg loss: {avg:.4f}')
    return avg


def eval_metrics(model, loader, stage='Eval'):
    """
    Compute metrics: accuracy, F1, ROC AUC, precision, recall, FNR, FPR.
    """
    model.eval()
    all_probs, all_labels = [], []
    for imgs, txts, mask, labels in tqdm(loader, desc=f'{stage} batches'):
        imgs, txts, mask = imgs.to(device), txts.to(device), mask.to(device)
        with torch.no_grad():
            _, _, _, probs = model(imgs, txts, text_mask=mask)
        all_probs.extend(probs.view(-1).cpu().numpy())
        all_labels.extend(labels.numpy().astype(int))
    # Convert to numpy arrays
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
        'FNR': fn / (fn + tp),  # False Negative Rate
        'FPR': fp / (fp + tn)   # False Positive Rate
    }
    print(f'[{stage}] ' + ', '.join([f'{k}: {v:.4f}' for k, v in stats.items()]))
    return stats


def run_training(df, hp_grid):
    """
    Main loop: iterate over hyperparameters, train, validate, and select best by F1.
    Saves metrics, best model, and loss plot to disk.
    """
    train_df, val_df, test_df = split_datasets(df)
    dim = train_df['img_embedding'].iloc[0].shape[-1]
    # Initialize best F1 tracking
    best_val_f1, best_cfg, best_losses = float('-inf'), None, None
    results = []  # Store validation metrics per config

    # Hyperparameter grid search
    for lr in hp_grid['lr']:
        for bs in hp_grid['batch_size']:
            for heads in hp_grid['num_heads']:
                for drop in hp_grid['dropout']:
                    for ff_r in hp_grid['ff_hidden_ratio']:
                        print(f'[Config] lr={lr}, bs={bs}, heads={heads}, drop={drop}, ff_r={ff_r}')
                        tr_loader = DataLoader(
                            EmbeddingPairDataset(train_df), batch_size=bs,
                            collate_fn=collate_fn, shuffle=True
                        )
                        va_loader = DataLoader(
                            EmbeddingPairDataset(val_df), batch_size=bs,
                            collate_fn=collate_fn
                        )
                        # Initialize model and optimizer
                        model = CrossModalBlock(dim, heads, drop, ff_r).to(device)
                        optimizer = optim.Adam(model.parameters(), lr=lr)
                        # Handle class imbalance with positive weight
                        wt = torch.tensor(train_df['label'].values, dtype=torch.float)
                        pos_weight = (wt == 0).sum() / (wt == 1).sum()
                        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
                        losses = []

                        # Training epochs
                        for e in range(1, hp_grid['epochs'] + 1):
                            print(f'-- Epoch {e}/{hp_grid["epochs"]}')
                            loss = train_one_epoch(model, tr_loader, optimizer, criterion)
                            losses.append(loss)
                            _ = eval_metrics(model, va_loader, stage='Val')

                        # Final validation metrics for this config
                        final_metrics = eval_metrics(model, va_loader, stage='Val final')
                        results.append({
                            **{'lr': lr, 'batch_size': bs, 'num_heads': heads,
                               'dropout': drop, 'ff_hidden_ratio': ff_r},
                            **final_metrics
                        })
                        # Update best model if F1 improved
                        if final_metrics['f1'] > best_val_f1:
                            best_val_f1 = final_metrics['f1']
                            best_model, best_cfg, best_losses = model, (lr, bs, heads, drop, ff_r), losses

    # Create directories and save results
    Path('metrics').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(
        Path('metrics') / 'validation_metrics.csv', index=False
    )
    print(f'[Test] Best config: {best_cfg} (by F1 score)')

    # Evaluate best model on test set and save
    test_loader = DataLoader(
        EmbeddingPairDataset(test_df), batch_size=best_cfg[1], collate_fn=collate_fn
    )
    test_stats = eval_metrics(best_model, test_loader, stage='Test')
    pd.DataFrame([test_stats]).to_csv(
        Path('metrics') / 'best_model_test_metrics.csv', index=False
    )
    torch.save(
        best_model.state_dict(), Path('models') / 'best_model_weights.pth'
    )

    # Plot and save training loss curve for best config
    plt.figure()
    plt.plot(range(1, len(best_losses) + 1), best_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Best Model Training Loss')
    plt.savefig(Path('metrics') / 'best_model_loss.png')

    return best_model

# Entry point for standalone execution
if __name__ == '__main__':
    # Load precomputed embeddings from pickle
    df = pd.read_pickle(Path.cwd() / 'results' / 'fusion_data.pkl')
    #df = pd.read_pickle('fusion_data.pkl')
    # Define hyperparameter grid for search
    hp_grid = {
        'lr': [1e-3,1e-6],
        'batch_size': [8,16],
        'num_heads': [8,16],
        'dropout': [0.1, 0.4],
        'ff_hidden_ratio': [2,6],
        'epochs': 3
    }
    # Run training and evaluation
    run_training(df, hp_grid)
    print('Done. Metrics, model weights, and loss plot saved.')