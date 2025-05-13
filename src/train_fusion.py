import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from fusion_model import HateClassifier


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
    test_loader,
    device,
    criterion,
    optimizer,
    checkpoint_dir,
    epochs=20,
    scheduler=None,
):
    num_training_steps = epochs * len(dataloader)
    progress_bar = tqdm(range(num_training_steps))

    print("training model")
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

            if scheduler:
                scheduler.step()
            progress_bar.update(1)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")
        log_to_file(
            f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%",
            os.path.join(checkpoint_dir, "training_log.txt"),
        )
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": total_loss,
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved to {checkpoint_path}")
        print(f"Evaluating model after epoch {epoch + 1}...")
        evaluate_model(model, test_loader, device, checkpoint_dir)


def log_to_file(message, filepath):
    with open(filepath, "a") as f:
        f.write(message + "\n")


def evaluate_model(model, dataloader, device, checkpoint_dir):
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
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    pos_acc = accuracy_score(all_labels[all_labels == 1], all_preds[all_labels == 1])
    neg_acc = accuracy_score(all_labels[all_labels == 0], all_preds[all_labels == 0])

    print("\n Evaluation Results:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Positive Accuracy: {pos_acc:.4f}")
    print(f"Negative Accuracy: {neg_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    log_to_file(
        f"Evaluation Results:\nAccuracy : {acc:.4f}\nPositive Accuracy: {pos_acc:.4f}\nNegative Accuracy: {neg_acc:.4f}\nPrecision: {precision:.4f}\nRecall   : {recall:.4f}\nF1 Score : {f1:.4f}",
        os.path.join(checkpoint_dir, "training_log.txt"),
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    print("loading data")
    root_dir = os.path.abspath("./NoHateZone")
    data_path = os.path.join(root_dir, "data/MMHS150K/fusion_data.pkl")
    checkpoint_dir = os.path.join(root_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # data_path = "../data/MMHS150K/fusion_data.pkl"  # TODO
    df = pd.read_pickle(data_path)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=19, stratify=df["label"]
    )
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    class_counts = train_df["label"].value_counts().to_dict()
    weights = train_df["label"].apply(lambda x: 1.0 / class_counts[x])
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    print("creating dataset")
    dataset = CrossModalDataset(train_df)
    test_dataset = CrossModalDataset(test_df)
    print("dataset size:", len(dataset))

    print("creating dataloader")
    dataloader = DataLoader(
        dataset, batch_size=32, sampler=sampler, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    print("creating model")
    model = HateClassifier(embed_dim=768).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-6)

    train_model(
        model,
        dataloader,
        test_dataloader,
        device,
        criterion,
        optimizer,
        checkpoint_dir,
        epochs=20,
    )

    final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    evaluate_model(model, test_dataloader, device, checkpoint_dir)


if __name__ == "__main__":
    main()
