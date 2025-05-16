from train_fusion import *


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    print("loading data")
    root_dir = os.path.abspath("./NoHateZone")
    data_path = os.path.join(root_dir, "data/HateMM/HateMM/embeddings_hatemm.pkl")

    checkpoint_dir = os.path.join(root_dir, "checkpoints_pretrained_MMH")
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

    test_dataset = CrossModalDataset(test_df)

    test_dataloader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    print("creating model")

    model = HateClassifier(embed_dim=768).to(device)
    pretrained_checkpoint = os.path.join(checkpoint_dir, "model_mmh_8.pt")
    checkpoint = torch.load(pretrained_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluate_model(model, test_dataloader, device, checkpoint_dir)


if __name__ == "__main__":
    main()
