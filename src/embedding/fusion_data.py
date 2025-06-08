import os

import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from utils import load_data


def get_labels(data):
    """extract labels from the data."""
    records = {}
    for tweet_id, info in data.items():
        labels = info.get("labels", [0, 0, 0])
        final_label = 0 if labels.count(0) >= 2 else 1
        records[tweet_id] = final_label
    return records


def load_embeddings(path):
    """Load embeddings from a .npy file."""
    return np.load(path, allow_pickle=True).item()


def get_clean_data(vit_embeddings_path, ocr_embeddings_path, labels_path):
    """Load and merge data from different sources to create a clean DataFrame with tweet IDs, image embeddings, text/ocr embeddings, and labels."""
    data = load_data(labels_path)
    vit_embeddings = load_embeddings(vit_embeddings_path)
    ocr_embeddings = load_embeddings(ocr_embeddings_path)
    labels = get_labels(data)

    vit_df = pd.DataFrame(
        {
            "tweet_id": list(vit_embeddings.keys()),
            "img_embedding": list(vit_embeddings.values()),
        }
    )
    ocr_df = pd.DataFrame(
        {
            "tweet_id": list(ocr_embeddings.keys()),
            "text_embedding": list(ocr_embeddings.values()),
        }
    )
    labels_df = pd.DataFrame(
        {"tweet_id": list(labels.keys()), "label": list(labels.values())}
    )

    vit_df["tweet_id"] = vit_df["tweet_id"].astype(int)
    ocr_df["tweet_id"] = ocr_df["tweet_id"].astype(int)
    labels_df["tweet_id"] = labels_df["tweet_id"].astype(int)

    merged_df = pd.merge(vit_df, ocr_df, on="tweet_id")
    final_df = pd.merge(merged_df, labels_df, on="tweet_id")

    return final_df


def main():
    root_path = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(root_path, "../../data/MMHS150K")
    vit_embeddings_path = os.path.join(root_dir, "vit_embeddings.npy")
    ocr_embeddings_path = os.path.join(root_dir, "tweet_ocr_embeddings.npy")
    labels_path = os.path.join(root_dir, "MMHS150K_GT.json")

    print("Loading data...")
    df = get_clean_data(vit_embeddings_path, ocr_embeddings_path, labels_path)

    print(f"Shape of the DataFrame: {df.shape}")

    df.to_pickle(os.path.join(root_dir, "fusion_data.pkl"))


if __name__ == "__main__":
    main()
