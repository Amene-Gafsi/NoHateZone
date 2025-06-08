import os
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from .crossval_model import HateClassifier


class CrossModalDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # [768]
        img_emb = torch.tensor(row["img_embedding"], dtype=torch.float32)

        # [num_tokens, 768]
        text_emb = torch.tensor(row["text_embedding"], dtype=torch.float32)

        # scalar
        label = torch.tensor(row["label"], dtype=torch.long)
        return text_emb, img_emb, label


def collate_fn(batch):
    text_seqs, img_vecs, labels = zip(*batch)

    text_seqs_padded = torch.nn.utils.rnn.pad_sequence(
        text_seqs, batch_first=True, padding_value=0.0
    )  # shape: [batch_size, max_len, 768]

    img_vecs = torch.stack(img_vecs)  # [batch_size, 768]
    img_vecs = img_vecs.unsqueeze(1)  # [batch_size, 1, 768]

    labels = torch.stack(labels)  # [batch_size]
    return text_seqs_padded, img_vecs, labels


def train_model(
    model,
    dataloader,
    device,
    criterion,
    optimizer,
    epochs=10,
):
    num_training_steps = epochs * len(dataloader)
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(epochs):
        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for text_emb, img_emb, labels in dataloader:
            text_emb = text_emb.to(device)
            img_emb = img_emb.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(text_emb, img_emb)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            progress_bar.update(1)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")


def log_to_file(message, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a") as f:
        f.write(message + "\n")


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for text_emb, img_emb, labels in dataloader:
            text_emb = text_emb.to(device)
            img_emb = img_emb.to(device)
            labels = labels.to(device)

            outputs = model(text_emb, img_emb)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    pos_acc = accuracy_score(all_labels[all_labels == 1], all_preds[all_labels == 1])
    neg_acc = accuracy_score(all_labels[all_labels == 0], all_preds[all_labels == 0])

    return acc, f1, pos_acc, neg_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    print("loading data")
    root_path = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(root_path, "../..")

    data_path = os.path.join(root_dir, "data/MMHS150K/fusion_data.pkl")
    checkpoint_dir = os.path.join(root_dir, "checkpoints/eval")
    os.makedirs(checkpoint_dir, exist_ok=True)

    df = pd.read_pickle(data_path)

    print(f"df size: {len(df)}")

    # cross validation
    weight_decays = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0]
    learning_rates = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    dropouts = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]

    num_heads = [2, 4, 8]
    fc_dims = [
        (512, 512, 128, 128, 32, 32),
        (512, 128, 32),
        (512, 256, 128, 64, 32, 16),
    ]
    arch = [0, 1]

    best_f1 = 0
    best_params = {}

    hyperparameter_combinations = product(
        weight_decays,
        learning_rates,
        dropouts,
        num_heads,
        fc_dims,
        arch,
    )
    for (
        weight_decay,
        lr,
        dropout,
        num_head,
        fc_dim,
        arch,
    ) in hyperparameter_combinations:
        print(
            f"Testing params: wd={weight_decay}, lr={lr}, dropout={dropout}, num_head={num_head}, fc_dim={fc_dim}, arch={arch}"
        )
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        f1_scores = []
        accuracy_scores = []
        pos_acc_scores = []
        neg_acc_scores = []

        X = df.index.values
        y = df["label"].values

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"Starting fold {fold + 1}")
            train_df_fold = df.iloc[train_idx]
            val_df_fold = df.iloc[val_idx]

            class_counts = train_df_fold["label"].value_counts().to_dict()
            weights = train_df_fold["label"].apply(lambda x: 1.0 / class_counts[x])
            weights = torch.DoubleTensor(weights.values)
            sampler = WeightedRandomSampler(
                weights, num_samples=len(weights), replacement=True
            )
            train_dataset = CrossModalDataset(train_df_fold)
            val_dataset = CrossModalDataset(val_df_fold)

            train_loader = DataLoader(
                train_dataset, batch_size=16, sampler=sampler, collate_fn=collate_fn
            )
            val_loader = DataLoader(
                val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn
            )
            model = HateClassifier(
                embed_dim=768,
                dropout=dropout,
                num_heads=num_head,
                fc_dims=fc_dim,
                arch=arch,
            ).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )

            train_model(
                model,
                train_loader,
                device,
                criterion,
                optimizer,
                epochs=2,
            )
            acc, f1, pos, neg = evaluate_model(model, val_loader, device)
            f1_scores.append(f1)
            accuracy_scores.append(acc)
            pos_acc_scores.append(pos)
            neg_acc_scores.append(neg)

        avg_f1 = np.mean(f1_scores)
        avg_acc = np.mean(accuracy_scores)
        avg_pos_acc = np.mean(pos_acc_scores)
        avg_neg_acc = np.mean(neg_acc_scores)
        log_msg = (
            f"Params: weight_decay={weight_decay}, lr={lr}, dropout={dropout}, num_head={num_head}, fc_dim={fc_dim}, arch={arch}\n"
            f"Average F1 Score: {avg_f1:.4f}\n"
            f"Average Accuracy: {avg_acc:.4f}\n"
            f"Average Positive Accuracy: {avg_pos_acc:.4f}\n"
            f"Average Negative Accuracy: {avg_neg_acc:.4f}\n\n"
        )

        print(log_msg)
        log_to_file(log_msg, os.path.join(checkpoint_dir, "cross_validation_log.txt"))

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_params = {
                "weight_decay": weight_decay,
                "lr": lr,
                "dropout": dropout,
                "num_head": num_head,
                "fc_dim": fc_dim,
                "arch": arch,
            }
    log_msg = (
        f"\nFinal Best F1 Score: {best_f1:.4f}\n"
        f"Final Best Parameters:\n"
        f"  Weight Decay: {best_params['weight_decay']}\n"
        f"  Learning Rate: {best_params['lr']}\n"
        f"  Dropout: {best_params['dropout']}\n"
        f"  Num Heads: {best_params['num_head']}\n"
        f"  FC Dims: {best_params['fc_dim']}\n"
        f"  Attention Architecture: {best_params['arch']}\n"
    )

    print(log_msg)
    log_to_file(log_msg, os.path.join(checkpoint_dir, "cross_validation_log.txt"))


if __name__ == "__main__":
    main()
