from train_fusion import *


def find_best_threshold(y_true, y_probs):
    best_f1 = 0
    best_thresh = 0.5

    thresholds = np.linspace(0.2, 0.8, 101)
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        print(f"Threshold: {t:.2f}, F1 Score: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh, best_f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading data")
    root_dir = os.path.abspath("./NoHateZone")
    data_path = os.path.join(root_dir, "data/HateMM/HateMM/embeddings_hatemm.pkl")
    checkpoint_dir = os.path.join(root_dir, "checkpoints")

    df = pd.read_pickle(data_path)
    dataset = CrossModalDataset(df)
    test_dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    print("Loading model")
    model = HateClassifier(embed_dim=768).to(device)
    pretrained_checkpoint = os.path.join(checkpoint_dir, "model_mmh_8.pt")
    checkpoint = torch.load(pretrained_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for text_emb, img_emb, labels in test_dataloader:
            text_emb = text_emb.to(device)
            img_emb = img_emb.to(device)
            labels = labels.to(device)

            outputs = model(text_emb, img_emb)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    best_thresh, best_f1 = find_best_threshold(all_labels, all_probs)
    print(f"\nBest threshold for F1: {best_thresh:.2f}, Best F1: {best_f1:.4f}")
    final_preds = (all_probs >= best_thresh).astype(int)

    precision = precision_score(all_labels, final_preds)
    recall = recall_score(all_labels, final_preds)
    acc = accuracy_score(all_labels, final_preds)
    auc = roc_auc_score(all_labels, all_probs)
    pos_acc = accuracy_score(all_labels[all_labels == 1], final_preds[all_labels == 1])
    neg_acc = accuracy_score(all_labels[all_labels == 0], final_preds[all_labels == 0])

    print("\nEvaluation with Best Threshold:")
    print(f"Accuracy         : {acc:.4f}")
    print(f"Positive Accuracy: {pos_acc:.4f}")
    print(f"Negative Accuracy: {neg_acc:.4f}")
    print(f"Precision        : {precision:.4f}")
    print(f"Recall           : {recall:.4f}")
    print(f"F1 Score         : {best_f1:.4f}")
    print(f"AUC              : {auc:.4f}")


if __name__ == "__main__":
    main()
