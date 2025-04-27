import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizer


def load_model(model_name="distilbert-base-uncased", device="cpu"):
    """Load the DistilBERT model and tokenizer."""
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def combine_text(tweet_text, ocr_text):
    """Concatenate tweet text and OCR text with a [SEP] token."""
    return f"{tweet_text} [SEP] {ocr_text}"


def embed_text(text, tokenizer, model, device="cpu"):
    """Generate embeddings for the given text."""
    encoded_input = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=512,
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    with torch.no_grad():
        output = model(**encoded_input)

    token_embeddings = output.last_hidden_state
    # sentence_embedding = token_embeddings.mean(dim=1)

    return token_embeddings.squeeze(0).cpu().numpy()  # (sequence_length, hidden_size)


def process_dataframe(df, tokenizer, model, device="cpu"):
    """Embed all rows from the DataFrame and return a dictionary {tweet_id: embedding}."""
    embeddings = {}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embedding rows"):
        tweet_id = row["tweet_id"]
        tweet_text = row.get("tweet_text", "")
        ocr_text = row.get("img_text", "")
        if pd.isna(tweet_text):
            tweet_text = ""
        if pd.isna(ocr_text):
            ocr_text = ""

        combined = combine_text(tweet_text, ocr_text)
        embedding = embed_text(combined, tokenizer, model, device)
        embeddings[tweet_id] = embedding

    return embeddings


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_dir = os.path.abspath("./NoHateZone/data/MMHS150K")  # <-- changed this line
    csv_path = os.path.join(root_dir, "tweet_ocr_dataset.csv")
    output_path = os.path.join(root_dir, "tweet_ocr_embeddings.npy")

    print("Loading model...")
    tokenizer, model = load_model()
    model.to(device)

    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    print("Processing dataset...")
    embeddings = process_dataframe(df, tokenizer, model, device)

    print("Saving embeddings...")
    np.save(output_path, embeddings)

    print(f"Saved embeddings to {output_path}")


if __name__ == "__main__":
    main()
