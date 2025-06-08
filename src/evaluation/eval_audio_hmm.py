import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
)


def find_best_threshold(df, model, tokenizer, device):
    all_labels = []
    all_probs = []

    model.eval()
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row["text"]) if pd.notna(row["text"]) else ""
        label = row["label"]

        inputs = tokenizer(text, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        label_tensor = torch.tensor([label]).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1)

        all_labels.append(label_tensor.item())
        all_probs.append(probs[:, 1].item())  # Probability of class 1

    best_f1 = 0
    best_thresh = 0.5

    thresholds = np.linspace(0.1, 0.8, 200)
    for t in thresholds:
        y_pred = (np.array(all_probs) >= t).astype(int)
        f1 = f1_score(all_labels, y_pred)
        print(f"Threshold: {t:.2f}, F1 Score: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh, best_f1


def evaluate_model(model, tokenizer, df, device, threshold=0.5):
    all_labels = []
    all_preds = []
    all_probs = []

    model.eval()
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row["text"]) if pd.notna(row["text"]) else ""
        label = row["label"]

        inputs = tokenizer(text, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        label_tensor = torch.tensor([label]).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1)
            prob_class1 = probs[:, 1].item()

        all_labels.append(label_tensor.item())
        all_probs.append(prob_class1)
        all_preds.append(int(prob_class1 >= threshold))

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    print("\nEvaluation Results:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")


def main():
    root_path = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(root_path, "../..")
    model_path = os.path.join(root_dir, "checkpoints/distilbert_hatespeech")
    dataset_path = os.path.join(root_dir, "data/labeled_audio_chunks.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    distilbert_model = DistilBertForSequenceClassification.from_pretrained(model_path)
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    distilbert_model.to(device)

    print("Loading data...")
    df = pd.read_csv(dataset_path)
    df = df[["text", "label"]].dropna()

    val_df, test_df = train_test_split(
        df, test_size=0.5, random_state=19, stratify=df["label"]
    )

    print("\nFinding best threshold...")
    best_thresh, best_f1 = find_best_threshold(
        val_df, distilbert_model, distilbert_tokenizer, device
    )
    print(f"\nBest threshold: {best_thresh:.4f}, F1: {best_f1:.4f}")

    print("\nEvaluating on test set with best threshold...")
    evaluate_model(
        distilbert_model, distilbert_tokenizer, test_df, device, threshold=best_thresh
    )


if __name__ == "__main__":
    main()
