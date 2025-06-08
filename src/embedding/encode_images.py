import json
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel


def load_data(json_path):
    """Load the JSON data."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def load_model(model_name, device):
    """Load the ViT model and feature extractor."""
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return feature_extractor, model


def extract_embedding(image_path, feature_extractor, model, device):
    """Extract the CLS token embedding from the image."""
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0]

    return cls_embedding.squeeze(0).cpu().numpy()


def process_dataset(data, img_dir, feature_extractor, model, device):
    """Process the dataset and extract embeddings for each image."""
    embeddings = {}
    failed = []

    for tweet_id, info in tqdm(data.items(), desc="Processing images"):
        img_path = os.path.join(img_dir, f"{tweet_id}.jpg")

        if not os.path.exists(img_path):
            failed.append(tweet_id)
            continue

        try:
            embedding = extract_embedding(img_path, feature_extractor, model, device)
            embeddings[tweet_id] = embedding
        except Exception as e:
            print(f"Failed on {tweet_id} because {e}")
            failed.append(tweet_id)

    return embeddings, failed


def save_embeddings(embeddings, output_path):
    """Save embeddings to a .npy file."""
    np.save(output_path, embeddings)


def load_embeddings(output_path):
    """Load embeddings from a .npy file."""
    return np.load(output_path, allow_pickle=True).item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_path = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(root_path, "../../data/MMHS150K")
    json_path = os.path.join(root_dir, "MMHS150K_GT.json")
    img_dir = os.path.join(root_dir, "img_resized")
    output_path = os.path.join(root_dir, "vit_embeddings.npy")

    model_name = "google/vit-base-patch16-224"

    print("Loading data...")
    data = load_data(json_path)
    print(f"Loaded {len(data)} images.")

    print("Loading model...")
    feature_extractor, model = load_model(model_name, device)

    print("Processing dataset...")
    embeddings, failed = process_dataset(
        data, img_dir, feature_extractor, model, device
    )

    print("Saving embeddings...")
    save_embeddings(embeddings, output_path)

    print(f"Saved {len(embeddings)} embeddings to {output_path}")
    print(f"Failed to process {len(failed)} images.")


if __name__ == "__main__":
    main()
