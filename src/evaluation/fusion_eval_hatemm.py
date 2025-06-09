import sys
import os

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "train")))

from train_fusion import *


def find_best_threshold(y_true, y_probs):
    best_f1 = 0
    best_thresh = 0.5

    thresholds = np.linspace(0.1, 0.8, 200)
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        print(f"Threshold: {t:.2f}, F1 Score: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh, best_f1


def get_probs_and_labels(dataloader, model, device):
    all_labels, all_probs = [], []
    with torch.no_grad():
        for text_emb, img_emb, labels in dataloader:
            text_emb = text_emb.to(device)
            img_emb = img_emb.to(device)
            labels = labels.to(device)

            outputs = model(text_emb, img_emb)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_probs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    print("loading data")

    root_path = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(root_path, "../..")
    data_path = os.path.join(root_dir, "data/HateMM/embeddings_hatemm.pkl")

    checkpoint_dir = os.path.join(root_dir, "checkpoints/hmm_modelseeds")

    df = pd.read_pickle(data_path)

    # train-test split on videos rather than frames same seed as training
    df["videoID"] = df["frameID"].apply(lambda x: "_".join(x.split("_")[:-1]))
    unique_videos = df["videoID"].unique()
    train_videos, temp_videos = train_test_split(
        unique_videos, test_size=0.2, random_state=19
    )

    val_videos, test_videos = train_test_split(
        temp_videos, test_size=0.5, random_state=21
    )

    # train_df = df[df["videoID"].isin(train_videos)].reset_index(drop=True)
    val_df = df[df["videoID"].isin(val_videos)].reset_index(drop=True)
    test_df = df[df["videoID"].isin(test_videos)].reset_index(drop=True)

    print(f"val size: {len(val_df)}, Test size: {len(test_df)}")

    val_dataset = CrossModalDataset(val_df)
    test_dataset = CrossModalDataset(test_df)

    val_dataloader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    print("creating model")
    model = HateClassifier(embed_dim=768).to(device)
    pretrained_checkpoint = os.path.join(checkpoint_dir, "model_epoch_16_seed1000.pt")
    checkpoint = torch.load(pretrained_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Finding best threshold on validation set...")
    val_labels, val_probs = get_probs_and_labels(val_dataloader, model, device)
    best_thresh, best_f1 = find_best_threshold(val_labels, val_probs)

    print(
        f"\nBest threshold from validation data: {best_thresh:.2f}, F1: {best_f1:.4f}"
    )

    print("\nEvaluating on held-out test set...")
    test_labels, test_probs = get_probs_and_labels(test_dataloader, model, device)
    final_preds = (test_probs >= best_thresh).astype(int)

    roc_df = pd.DataFrame({"true_label": test_labels, "predicted_prob": test_probs})
    roc_df.to_csv(os.path.join(checkpoint_dir, "roc_1000.csv"), index=False)
    print("Saved ROC data to roc_data_epoch16.csv")

    precision = precision_score(test_labels, final_preds)
    recall = recall_score(test_labels, final_preds)
    f1 = f1_score(test_labels, final_preds)
    macro_f1 = f1_score(test_labels, final_preds, average="macro")
    acc = accuracy_score(test_labels, final_preds)
    auc = roc_auc_score(test_labels, test_probs)
    pos_acc = accuracy_score(
        test_labels[test_labels == 1], final_preds[test_labels == 1]
    )
    neg_acc = accuracy_score(
        test_labels[test_labels == 0], final_preds[test_labels == 0]
    )

    print("\nEvaluation on Final Test Set (using best threshold from validation):")
    print(f"Accuracy         : {acc:.4f}")
    print(f"Positive Accuracy: {pos_acc:.4f}")
    print(f"Negative Accuracy: {neg_acc:.4f}")
    print(f"Precision        : {precision:.4f}")
    print(f"Recall           : {recall:.4f}")
    print(f"F1 Score         : {f1:.4f}")
    print(f"Macro F1 Score   : {macro_f1:.4f}")
    print(f"AUC              : {auc:.4f}")


if __name__ == "__main__":
    main()
