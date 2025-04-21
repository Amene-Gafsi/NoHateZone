import json
import os
import re

import pandas as pd
from IPython.display import display

from utils import load_data


def clean_tweet(text):
    """Remove URLs, @mentions, and emojis from tweet text."""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s,.!?]", "", text)
    text = text.strip()
    return text


def process_data(data, img_txt_dir):
    """Process the tweets and extract cleaned tweet and OCR text."""
    records = []
    for tweet_id, info in data.items():
        tweet_text = info.get("tweet_text", "No tweet text available")
        clean_text = clean_tweet(tweet_text)

        ocr_path = os.path.join(img_txt_dir, f"{tweet_id}.json")
        img_text = ""

        if os.path.exists(ocr_path):
            with open(ocr_path, "r", encoding="utf-8") as f:
                ocr_data = json.load(f)
                img_text = ocr_data.get("img_text", "")

        records.append(
            {"tweet_id": tweet_id, "tweet_text": clean_text, "img_text": img_text}
        )

    return pd.DataFrame(records)


def save_dataframe(df, output_path):
    """Save the DataFrame to a CSV file."""
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned DataFrame to {output_path}")


def main():
    root_dir = "../data"
    json_path = os.path.join(root_dir, "MMHS150K_GT.json")
    img_txt_dir = os.path.join(root_dir, "img_txt")
    output_csv_path = os.path.join(root_dir, "tweet_ocr_dataset.csv")

    print("Loading data...")
    data = load_data(json_path)

    print("Processing data...")
    df = process_data(data, img_txt_dir)

    display(df)

    print("Saving DataFrame...")
    save_dataframe(df, output_csv_path)


if __name__ == "__main__":
    main()
