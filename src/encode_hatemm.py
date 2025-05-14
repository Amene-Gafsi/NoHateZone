import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import encode_images as ei
import encode_tweets as et
from utils import extract_frames


def extract_frames_from_videos(HateMM_dir):
    # Extract frames from hate videos
    hate_videos_dir = os.path.join(HateMM_dir, "hate_videos", "hate_videos")
    non_hate_videos_dir = os.path.join(HateMM_dir, "non_hate_videos", "non_hate_videos")
    hate_frames_dir = os.path.join(HateMM_dir, "hate_frames")
    non_hate_frames_dir = os.path.join(HateMM_dir, "non_hate_frames")
    print("Extracting frames from hate videos...")
    for filename in os.listdir(hate_videos_dir):
        if filename.lower().endswith(".mp4"):
            video_name = filename.split(".")[0]
            video_index = video_name.split("_")[-1]
            extract_frames(
                os.path.join(hate_videos_dir, filename),
                hate_frames_dir,
                video_index=video_index,
            )
    print("Extracting frames from non-hate videos...")
    for filename in os.listdir(non_hate_videos_dir):
        if filename.lower().endswith(".mp4"):
            video_name = filename.split(".")[0]
            video_index = video_name.split("_")[-1]
            extract_frames(
                os.path.join(non_hate_videos_dir, filename),
                non_hate_frames_dir,
                video_index=video_index,
            )


def process_hatemm_frames(HateMM_dir, feature_extractor, model, device):
    embeddings = {}
    failed = []
    hate_frames_dir = os.path.join(HateMM_dir, "hate_frames")
    non_hate_frames_dir = os.path.join(HateMM_dir, "non_hate_frames")

    for filename in tqdm(os.listdir(hate_frames_dir)):
        if filename.lower().endswith(".jpg"):
            frame_ID = "hate_" + filename
            img_path = os.path.join(hate_frames_dir, filename)
            try:
                embedding = ei.extract_embedding(
                    img_path, feature_extractor, model, device
                )
                embeddings[frame_ID] = embedding
            except Exception as e:
                print(f"Failed on {filename} because {e}")
                failed.append(filename)

    for filename in tqdm(os.listdir(non_hate_frames_dir)):
        if filename.lower().endswith(".jpg"):
            frame_ID = "non_hate_" + filename
            img_path = os.path.join(non_hate_frames_dir, filename)
            try:
                embedding = ei.extract_embedding(
                    img_path, feature_extractor, model, device
                )
                embeddings[frame_ID] = embedding
            except Exception as e:
                print(f"Failed on {filename} because {e}")
                failed.append(filename)

    return embeddings, failed


def encode_frames(HateMM_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_path = os.path.join(HateMM_dir, "frames_embedding.npy")

    model_name = "google/vit-base-patch16-224"

    print("Loading model...")
    feature_extractor, model = ei.load_model(model_name, device)

    print("Encoding frames...")
    embeddings, failed = process_hatemm_frames(
        HateMM_dir, feature_extractor, model, device
    )

    print("Saving embeddings...")
    ei.save_embeddings(embeddings, output_path)

    print(f"Saved {len(embeddings)} embeddings to {output_path}")
    print(f"Failed to process {len(failed)} images.")
    return embeddings


def process_hatemm_ocr(df, tokenizer, model, device):
    embeddings = {}
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embedding rows"):
        frameID = row["frame_file"]
        ocr_text = row.get("text", "")
        if pd.isna(ocr_text):
            ocr_text = ""

        embedding = et.embed_text(ocr_text, tokenizer, model, device)
        embeddings[frameID] = embedding

    return embeddings


def encode_ocr(HateMM_dir, df):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_path = os.path.join(HateMM_dir, "ocr_embeddings.npy")

    print("Loading model...")
    tokenizer, model = et.load_model()
    model.to(device)

    print("Processing dataset...")
    embeddings = process_hatemm_ocr(df, tokenizer, model, device)

    print("Saving embeddings...")
    np.save(output_path, embeddings)

    print(f"Saved embeddings to {output_path}")
    return embeddings


def merge_embeddings(frame_embeddings, ocr_embeddings, df):
    print("Merging embeddings...")
    df["img_embedding"] = df["frame_file"].map(frame_embeddings)
    df["text_embedding"] = df["frame_file"].map(ocr_embeddings)
    df_clean = df.dropna(subset=["img_embedding", "text_embedding", "Frame_hate"])
    merged_df = df_clean[
        ["frame_file", "img_embedding", "text_embedding", "Frame_hate"]
    ].copy()
    merged_df.rename(
        columns={"frame_file": "frameID", "Frame_hate": "label"}, inplace=True
    )
    return merged_df


def main():
    hateMM_dir = "./NoHateZone/data/HateMM/HateMM/"
    df = pd.read_csv(os.path.join(hateMM_dir, "processed_hatemm.csv"))

    extract_frames_from_videos(hateMM_dir)
    frame_embeddings = encode_frames(hateMM_dir)

    ocr_embeddings = encode_ocr(hateMM_dir, df)

    final_df = merge_embeddings(frame_embeddings, ocr_embeddings, df)
    print(f"Shape of the DataFrame: {df.shape}")
    final_df.to_pickle(os.path.join(hateMM_dir, "embeddings_hatemm.pkl"))


if __name__ == "__main__":
    main()
