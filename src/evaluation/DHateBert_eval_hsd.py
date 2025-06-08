import os

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
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

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train')))
from bert_finetune import get_train_test_datasets, load_data


def evaluate_model(model, test_loader, device):
    """
    Evaluates a text classification model's performance on a test dataset.

    Parameters:
    - model (torch.nn.Module): The model to be evaluated.
    - test_loader (torch.utils.data.DataLoader): A DataLoader containing the test dataset.
    - device (torch.device): The device to run the evaluation on.
    - checkpoint_dir (str): Directory path to save the evaluation log.

    Returns:
    - all_preds (torch.Tensor): Predictions made by the model.
    - all_labels (torch.Tensor): Ground truth labels.
    - acc (float): Accuracy score.
    - f1 (float): F1 score.
    """
    all_labels = []
    all_preds = []
    all_probs = []

    model.eval()

    for b in tqdm(test_loader):
        inputs = {k: v.to(device) for k, v in b.items() if k != "labels"}
        labels = b["labels"].to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1)  # Probabilities
            preds = torch.argmax(probs, dim=-1)

        all_labels.extend(labels.cpu().view(-1).numpy())
        all_preds.extend(preds.cpu().view(-1).numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
        # Assuming binary classification

    # Convert to tensors
    all_labels = torch.tensor(all_labels)
    all_preds = torch.tensor(all_preds)
    all_probs = torch.tensor(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    pos_acc = accuracy_score(all_labels[all_labels == 1], all_preds[all_labels == 1])
    neg_acc = accuracy_score(all_labels[all_labels == 0], all_preds[all_labels == 0])
    auc = roc_auc_score(all_labels, all_probs)
    print("\n Evaluation Results:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Positive Accuracy: {pos_acc:.4f}")
    print(f"Negative Accuracy: {neg_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")
    print(classification_report(all_labels, all_preds))

    return all_preds, all_labels, acc, f1


def main():
    root_path = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(root_path, "../..") 
    
    data_path = os.path.join(root_dir, "data/HateSpeechDataset/HateSpeechDataset.csv")
    model_path = os.path.join(
        root_dir, "checkpoints/distilbert_hatespeech"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    distilbert_model = DistilBertForSequenceClassification.from_pretrained(model_path)
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_path)

    print("loading data")
    X_train, X_test, y_train, y_test = load_data(data_path)

    train_data, test_data = get_train_test_datasets(
        X_train, y_train, X_test, y_test, distilbert_tokenizer
    )
    distilbert_model.to(device)
    print("Evaluating model...")
    evaluate_model(distilbert_model, train_data, device)


if __name__ == "__main__":
    main()
