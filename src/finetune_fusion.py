from torch.optim.lr_scheduler import OneCycleLR

from train_fusion import *

# TODO
# train with hatemm train-test split on videos not frames / crossval on lr and wd
# log all performace metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    print("loading data")
    root_dir = os.path.abspath("./NoHateZone")
    data_path = os.path.join(root_dir, "data/HateMM/HateMM/embeddings_hatemm.pkl")

    checkpoint_dir = os.path.join(root_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    df = pd.read_pickle(data_path)

    # train-test split on videos rather than frames
    df["videoID"] = df["frameID"].apply(lambda x: "_".join(x.split("_")[:-1]))
    unique_videos = df["videoID"].unique()
    train_videos, test_videos = train_test_split(
        unique_videos, test_size=0.15, random_state=19
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
    pretrained_checkpoint = os.path.join(checkpoint_dir, "model_epoch_8.pt")
    checkpoint = torch.load(pretrained_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-4,
        steps_per_epoch=len(dataloader),
        epochs=20,
    )

    checkpoint_dir = os.path.join(root_dir, "checkpoints_finetune")
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

    final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    evaluate_model(model, test_dataloader, device, checkpoint_dir)


if __name__ == "__main__":
    main()
