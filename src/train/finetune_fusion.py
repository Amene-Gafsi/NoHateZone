from torch.optim.lr_scheduler import OneCycleLR
import random
import numpy as np
import torch

from train_fusion import *


def eval_model(model, dataloader, device, checkpoint_dir):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for text_emb, img_emb, labels in dataloader:
            text_emb = text_emb.to(device)
            img_emb = img_emb.to(device)
            labels = labels.to(device)

            outputs = model(text_emb, img_emb)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, dim=1)
            # _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

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
    log_to_file(
        f"Evaluation Results:\nAccuracy : {acc:.4f}\nPositive Accuracy: {pos_acc:.4f}\nNegative Accuracy: {neg_acc:.4f}\nPrecision: {precision:.4f}\nRecall   : {recall:.4f}\nF1 Score : {f1:.4f}\nAUC : {auc:.4f}",
        os.path.join(checkpoint_dir, "training_log.txt"),
    )


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(1000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    print("loading data")
    root_path = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(root_path, "../..")

    data_path = os.path.join(root_dir, "data/HateMM/HateMM/embeddings_hatemm.pkl")

    checkpoint_dir = os.path.join(root_dir, "checkpoints/pretrained_MMH")
    os.makedirs(checkpoint_dir, exist_ok=True)

    df = pd.read_pickle(data_path)

    # train-test split on videos rather than frames
    df["videoID"] = df["frameID"].apply(lambda x: "_".join(x.split("_")[:-1]))
    unique_videos = df["videoID"].unique()
    train_videos, test_videos = train_test_split(
        unique_videos, test_size=0.2, random_state=19
    )
    train_df = df[df["videoID"].isin(train_videos)].reset_index(drop=True)
    test_df = df[df["videoID"].isin(test_videos)].reset_index(drop=True)

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
    pretrained_checkpoint = os.path.join(checkpoint_dir, "model_mmh.pt")
    checkpoint = torch.load(pretrained_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-6,
        steps_per_epoch=len(dataloader),
        epochs=20,
    )

    checkpoint_dir = os.path.join(root_dir, "checkpoints/finetuned_HMM")
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_model(
        model,
        dataloader,
        test_dataloader,
        device,
        criterion,
        optimizer,
        checkpoint_dir,
        epochs=20,
        scheduler=scheduler,
    )

    final_model_path = os.path.join(checkpoint_dir, "model_hatemm.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    eval_model(model, test_dataloader, device, checkpoint_dir)


if __name__ == "__main__":
    main()
