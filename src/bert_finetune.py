import os
import random

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Sampler
from torch.utils.data import Dataset as TorchDataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)


class HateSpeechDataset(TorchDataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True)

        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx]),
        }


class CustomDataSampler(Sampler):
    def __init__(
        self,
        dataset,
        sampling_weights,
    ):
        """
        sampling_weights: dictionary, where keys are classes and values
        are weights of each class, not necessarily summing up to one.

        dataset: imbalanced dataset
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.number_of_classes = len(sampling_weights)
        self.sampling_weights = sampling_weights

        self.class_indices = {}
        for idx in range(len(dataset)):
            item = dataset[idx]
            label = int(item["labels"].item())

            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

        self.num_samples_per_class = {}
        for cls, indices in self.class_indices.items():
            weight = self.sampling_weights.get(cls, 1)
            self.num_samples_per_class[cls] = int(len(indices) * weight)

    def __iter__(self):
        sampled_indices = []

        for cls, num_samples in self.num_samples_per_class.items():
            indices = self.class_indices[cls]
            if len(indices) == 0:
                continue
            sampled_indices.extend(random.choices(indices, k=num_samples))

        random.shuffle(sampled_indices)
        return iter(sampled_indices)

    def __len__(self):
        # return len of the resampled dataset
        return sum(self.num_samples_per_class.values())


def load_pretrained(model_name, num_labels=2, device=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    if device:
        model = model.to(device)
    return tokenizer, model


def load_data(file_path):
    df = pd.read_csv(file_path)
    df.drop(columns=["Content_int"], inplace=True)

    df = df[df["Label"].isin(["0", "1"])]
    df = df.reset_index(drop=True)
    df["Label"] = df["Label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df["Content"].tolist(), df["Label"].tolist(), test_size=0.15, random_state=42
    )
    return X_train, X_test, y_train, y_test


def get_train_test_datasets(X_train, y_train, X_test, y_test, tokenizer):
    train_data = HateSpeechDataset(X_train, y_train, tokenizer)
    test_data = HateSpeechDataset(X_test, y_test, tokenizer)
    return train_data, test_data


def train_model(
    model,
    train_loader,
    optimizer,
    lr_scheduler,
    num_epochs,
    device,
    save_path_prefix="checkpoints/distilbert_hatespeech",
):
    os.makedirs(save_path_prefix, exist_ok=True)
    log_file = "checkpoints/distilbert_hatespeech/training_log.txt"

    num_training_steps = num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            epoch_loss += loss.item()

        torch.save(model.state_dict(), f"{save_path_prefix}_{epoch + 1}.pth")
        log_message = f"[Epoch {epoch + 1}] Loss: {epoch_loss:.4f} | Saved"
        log_to_file(log_message, log_file)
        print(log_message)


def evaluate_model(model, test_loader, device):
    """
    Evaluates a text classification model's performance on a test dataset.
    Parameters:
    - model (torch.nn.Module): The model to be evaluated.
    - tokenizer (transformers.PreTrainedTokenizer):
        The tokenizer used to process the text data into a format that the model can
        understand.
    - test_loader (torch.utils.data.DataLoader): A DataLoader containing the test
        dataset.
    Returns:
    - all_preds (torch.Tensor): A tensor containing all the predictions made by the
        model on the test dataset.
    - all_labels (torch.Tensor): A tensor containing all the labels in the test dataset.
    - acc (float): The overall accuracy of the model on the test dataset.
    - f1 (float): The F1 score of the model on the test dataset.
    """
    all_labels = None
    all_preds = None
    model.eval()

    for b in tqdm(test_loader):
        inputs = {k: v.to(device) for k, v in b.items() if k != "labels"}
        labels = b["labels"].to(device)

        # run the model to make the prediction
        with torch.no_grad():
            pred = model(**inputs).logits.argmax(dim=-1)

        if all_labels is None:
            all_labels = labels.cpu()
            all_preds = pred.cpu()
        else:
            all_labels = torch.concat([all_labels, labels.cpu()])
            all_preds = torch.concat([all_preds, pred.cpu()])

    assert len(all_preds) == len(all_labels), "Test Failed. Check your code!"
    # compute f1 score between model predictions and ground-truth labels (you can use sklearn.metrics)
    f1 = f1_score(all_labels, all_preds)

    # compute accuracy score between model predictions and ground-truth labels (you can use sklearn.metrics)
    acc = accuracy_score(all_labels, all_preds)

    # compute the accuracy on Positive(label==1) samples
    pos_acc = accuracy_score(all_labels[all_labels == 1], all_preds[all_labels == 1])

    # compute the accuracy on Negative(label==0) samples
    neg_acc = accuracy_score(all_labels[all_labels == 0], all_preds[all_labels == 0])

    print("Accuracy: ", acc * 100, "%")
    print(" -- Positive Accuracy: ", pos_acc * 100, "%")
    print(" -- Negative Accuracy: ", neg_acc * 100, "%")
    print("F1 score: ", f1)

    return all_preds, all_labels, acc, f1


def log_to_file(message, filepath="checkpoints/training_log.txt"):
    with open(filepath, "a") as f:
        f.write(message + "\n")


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load DistilBert (https://huggingface.co/distilbert/distilbert-base-uncased)
    distilbert_tokenizer, distilbert_model = load_pretrained(
        "distilbert-base-uncased", num_labels=2, device=device
    )

    # Load data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "data", "HateSpeechDataset.csv")
    data_path = os.path.abspath(data_path)

    X_train, X_test, y_train, y_test = load_data(data_path)

    train_data, test_data = get_train_test_datasets(
        X_train, y_train, X_test, y_test, distilbert_tokenizer
    )

    sampling_weights = {0: 1, 1: y_train.count(0) / y_train.count(1)}
    train_sampler = CustomDataSampler(train_data, sampling_weights=sampling_weights)

    train_loader = DataLoader(train_data, batch_size=32, sampler=train_sampler)
    test_loader = DataLoader(test_data, batch_size=32)

    optimizer = AdamW(distilbert_model.parameters(), lr=5e-5)
    num_epochs = 10

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=1,
        num_training_steps=num_epochs * len(train_loader),
    )

    # train the model
    train_model(
        distilbert_model, train_loader, optimizer, lr_scheduler, num_epochs, device
    )

    all_preds, all_labels, acc, f1 = evaluate_model(
        distilbert_model, test_loader, device
    )
    # Save the model
    distilbert_model.save_pretrained("checkpoints/distilbert_hatespeech")
    distilbert_tokenizer.save_pretrained("checkpoints/distilbert_hatespeech")

    log_to_file(
        f"[Evaluation] Accuracy: {acc * 100:.2f}%, F1: {f1:.4f}",
        "checkpoints/training_log.txt",
    )


if __name__ == "__main__":
    main()
