import json
import os
import re
import shutil
import subprocess
import sys
import textwrap

sys.modules["flash_attn"] = None
sys.modules["flash_attn_2_cuda"] = None

import sys

import audio_extract
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from pydub import AudioSegment
from pydub.generators import Sine
from tabulate import tabulate
from transformers import (
    AutoModel,
    AutoTokenizer,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
)

from .audio2text import AudioTranscriber

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "embedding"))
)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
)

from encode_images import load_model as load_image_model
from encode_tweets import load_model as load_tweet_model
from fusion_model import HateClassifier


def extract_audio(input_path, output_dir):
    """Extract audio from video file defined by input_path and save it to output_path.
    Args:
        input_path (str): Path to the input video file.
        output_dir (str): Path to the output directory where audio will be saved.
    """

    os.makedirs(output_dir, exist_ok=True)
    # base_name = os.path.splitext(os.path.basename(input_path))[0]
    # output_path = os.path.join(output_dir, base_name + ".mp3")
    output_path = os.path.join(output_dir, "audio.mp3")

    if os.path.exists(output_path):
        os.remove(output_path)

    try:
        audio = audio_extract.extract_audio(input_path, output_path)
        return audio
    except Exception as e:
        print(f"[WARNING] No audio found or extraction failed: {e}")
        silent_audio = AudioSegment.silent(duration=1000)  # 1 second of silence
        silent_audio.export(output_path, format="mp3")
        print(f"Saved silent audio at {output_path}")
        return output_path


def extract_frames(input_path, output_dir, video_index=None, frequency=30):
    """
    Extract frames from video file and save them to output_dir.
    Records frame number and timestamp in a DataFrame.

    Args:
        input_path (str): Path to the input video file.
        output_dir (str): Path to the output directory where frames will be saved.
        frequency (int): Frequency of frames to save (e.g., save every nth frame).
        video_index (int): Index of the video to include in frame filenames.

    Returns:
        tuple: (number of frames extracted, DataFrame with 'frame_number' and 'timestamp_seconds' columns)
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    frame_count = 0
    success = True
    records = []

    while success:
        success, image = cap.read()
        if not success:
            break

        if count % frequency == 0:
            timestamp = count / fps
            if video_index is not None:
                filename = os.path.join(
                    output_dir, f"video_{video_index}_frame{count}.jpg"
                )
            else:
                filename = os.path.join(output_dir, f"frame{count}.jpg")
            cv2.imwrite(filename, image)
            records.append((count, timestamp))
            frame_count += 1

        count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {input_path}, saved to {output_dir}.")

    df = pd.DataFrame(records, columns=["frame_number", "timestamp_seconds"])
    return frame_count, df


def create_chunk_dataframe(chunk_info):
    """
    Create a DataFrame from a list of dictionaries containing chunk information,
    combine 'start' and 'end' into a single 'timestamp' column, and return the DataFrame.

    Parameters:
    -----------
    chunk_info : list of dict
        Each dictionary should contain at least 'start', 'end', and 'text' keys.

    Returns:
    --------
    pd.DataFrame
        DataFrame with two columns: 'timestamp' and 'text'.
        The 'timestamp' column holds a list [start, end] for each chunk.
    """
    # Create the DataFrame from the provided list of dictionaries
    df = pd.DataFrame(chunk_info)

    # Create a new column 'timestamp' that holds [start, end] as a list
    df["timestamp"] = df.apply(lambda row: [row["start"], row["end"]], axis=1)

    # Drop the original 'start' and 'end' columns
    df = df.drop(columns=["start", "end"])

    # Reorder columns to have 'timestamp' first and then 'text'
    df = df[["timestamp", "text"]]

    return df


def combine_text(chunk_text, frame_text):
    """
    Combine the existing chunk text and frame text with ' | ' in between,
    but only if both sides are non-empty after stripping whitespace.
    """
    chunk_str = str(chunk_text).strip()
    frame_str = str(frame_text).strip()

    # If the chunk text is empty, return it unchanged (do nothing).
    if chunk_str == "":
        return chunk_str

    # If the chunk text is not empty but the frame text is empty, also do nothing.
    if frame_str == "":
        return chunk_str

    # Both are non-empty; combine them with ' | '.
    return chunk_str + " | " + frame_str


def load_data(json_path):
    """Load the JSON data."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def normalize(word):
    """Normalize a word by removing non-alphanumeric characters and converting to lowercase."""
    return re.sub(r"[^\w]", "", word).lower()


def extract_audio_and_frames(video_path, audio_path, frames_path):
    """Prepare audio and frames for processing. and return the path to the audio file path."""
    extract_audio(video_path, audio_path)
    extract_frames(video_path, frames_path, frequency=30)
    audio_files = [
        f
        for f in os.listdir(audio_path)
        if f.endswith((".mp3", ".wav", ".aac", ".flac", ".m4a"))
    ]
    if len(audio_files) != 1:
        raise Exception(
            f"Expected exactly one audio file, but found {len(audio_files)}."
        )
    return os.path.join(audio_path, audio_files[0])


def transcribe_audio(input_audio, device):
    """Transcribe the audio file using the AudioTranscriber class and return the transcription."""
    transcriber = AudioTranscriber(device=device)
    return transcriber.transcribe_audio(
        input_audio=input_audio, return_timestamps="word"
    )


def classify_sentences(text, model_path):
    """Split the transcribtion into sentences and classify them using finetuned DistilBERT. Returns a DataFrame sentences and labels et probability of each class."""
    sentences = [
        sentence.strip()
        for sentence in re.findall(r"[^.!?]*[.!?]", text)
        if sentence.strip()
    ]

    labels = []
    distilbert_model = DistilBertForSequenceClassification.from_pretrained(model_path)
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    distilbert_model.eval()

    for sentence in sentences:
        inputs = distilbert_tokenizer(
            sentence, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = distilbert_model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).squeeze()
            predicted_class = torch.argmax(probs).item()
            prob_class_0 = round(probs[0].item(), 3)
            prob_class_1 = round(probs[1].item(), 3)
        labels.append(
            {
                "sentence": sentence,
                "label": predicted_class,
                "prob_class_0": prob_class_0,
                "prob_class_1": prob_class_1,
            }
        )
    df = pd.DataFrame(labels)
    df_display = df[["sentence", "label", "prob_class_1"]].copy()
    df_display["sentence"] = df_display["sentence"].apply(
        lambda s: wrap_text(s, width=60)
    )

    print(tabulate(df_display, headers="keys", tablefmt="grid", showindex=False))

    return df


def convert_to_min_sec(beep_intervals):
    """Convert list of (start, end) intervals in seconds to (min:sec) format."""
    converted = []
    for start, end in beep_intervals:
        start_min = int(start // 60)
        start_sec = int(start % 60)
        end_min = int(end // 60)
        end_sec = int(end % 60)
        converted.append((f"{start_min}:{start_sec:02d}", f"{end_min}:{end_sec:02d}"))
    return converted


def find_beep_intervals(df, transcription):
    """Find the intervals in the transcription where the beep should be added based on the classified sentences."""
    beep_intervals = []

    transcript_words = [
        {
            "word": normalize(chunk["text"]),
            "start": chunk["timestamp"][0],
            "end": chunk["timestamp"][1],
        }
        for chunk in transcription.get("chunks", [])
    ]
    transcript_word_list = [w["word"] for w in transcript_words]

    for idx, row in df.iterrows():
        if row["label"] != 1:
            continue

        sentence = row["sentence"]
        sentence_words = [normalize(w) for w in sentence.split() if normalize(w)]

        if not sentence_words:
            continue

        for i in range(len(transcript_word_list) - len(sentence_words) + 1):
            window = transcript_word_list[i : i + len(sentence_words)]
            if window == sentence_words:
                start_time = transcript_words[i]["start"]
                end_time = transcript_words[i + len(sentence_words) - 1]["end"]
                if (end_time - start_time) <= 0.5:
                    continue
                beep_intervals.append((start_time, end_time))
                print(
                    f"[MATCH] '{sentence}' → BEEP from {start_time:.2f}s to {end_time:.2f}s"
                )
    beep_intervals = clean_beep_intervals(beep_intervals)
    print("Beeping :", convert_to_min_sec(beep_intervals))
    return beep_intervals


def clean_beep_intervals(beep_intervals):
    """Remove duplicates, sort, and merge overlapping intervals."""

    # Remove duplicates and sort
    intervals = sorted(set(beep_intervals))

    merged = []
    for start, end in intervals:
        if not merged:
            merged.append((start, end))
        else:
            last_start, last_end = merged[-1]
            if start <= last_end:  # Overlapping or touching intervals
                merged[-1] = (last_start, max(last_end, end))  # Merge them
            else:
                merged.append((start, end))
    return merged


def censor_audio(input_audio_path, beep_intervals, output_audio_path):
    """Censor the audio by adding beeps to the specified intervals."""
    audio = AudioSegment.from_file(input_audio_path)
    censored_audio = AudioSegment.empty()
    current_pos = 0

    for start_sec, end_sec in sorted(beep_intervals):
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        duration = end_ms - start_ms
        censored_audio += audio[current_pos:start_ms]
        beep = Sine(1000).to_audio_segment(duration=duration).apply_gain(-3.0)
        beep = beep.set_frame_rate(audio.frame_rate)
        censored_audio += beep
        current_pos = end_ms

    censored_audio += audio[current_pos:]
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    censored_audio.export(output_audio_path, format="wav")
    print("Censored audio saved as", output_audio_path)


def load_all_models(device, hate_classifier_path="../checkpoints/model_epoch_1.pt"):
    """Load all models required for the pipeline
    returns:
        feature_extractor: Feature extractor for image model
        model_img: Image model
        tokenizer_tweet: Tokenizer for tweet model
        model_tweet: Tweet model
        tokenizer_ocr: Tokenizer for OCR model
        model_ocr: OCR model
        hate_classifier: Hate classifier model
    """
    feature_extractor, model_img = load_image_model(
        "google/vit-base-patch16-224", device=device
    )
    tokenizer_tweet, model_tweet = load_tweet_model(
        model_name="distilbert-base-uncased", device=device
    )
    tokenizer_ocr = AutoTokenizer.from_pretrained(
        "ucaslcl/GOT-OCR2_0", trust_remote_code=True
    )
    model_ocr = (
        AutoModel.from_pretrained(
            "ucaslcl/GOT-OCR2_0",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=device,
            use_safetensors=True,
            pad_token_id=tokenizer_ocr.eos_token_id,
        )
        .to(device)
        .eval()
    )

    hate_classifier = HateClassifier(embed_dim=768).to(device)
    checkpoint = torch.load(hate_classifier_path, map_location=device)
    hate_classifier.load_state_dict(checkpoint["model_state_dict"])
    # hate_classifier.load_state_dict(checkpoint)
    hate_classifier.to(device).eval()

    return (
        feature_extractor,
        model_img,
        tokenizer_tweet,
        model_tweet,
        tokenizer_ocr,
        model_ocr,
        hate_classifier,
    )


def extract_frame_number(filename):
    numbers = re.findall(r"\d+", filename)
    return int(numbers[-1]) if numbers else -1


def get_image_embedding(image_path, feature_extractor, model_img, device):
    """Get the image embedding from the image model."""
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model_img(**inputs)
    image_embedding = outputs.last_hidden_state[:, 0]  # CLS token
    image_embedding = image_embedding.squeeze(0)  # shape: [768]
    return image_embedding.unsqueeze(0).unsqueeze(1).to(device)  # [1, 1, 768]


def get_text_embedding(
    image_path, tokenizer_ocr, model_ocr, tokenizer_tweet, model_tweet, device
):
    """Use OCR to get the text from the image and then get the text embedding."""
    ocr_result = model_ocr.chat(tokenizer_ocr, image_path, ocr_type="ocr")
    print("Frame ocr result:", ocr_result)

    encoded_input = tokenizer_tweet(
        ocr_result,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=512,
    ).to(device)
    with torch.no_grad():
        outputs = model_tweet(**encoded_input)
    text_embedding = outputs.last_hidden_state.squeeze(0)
    return text_embedding.unsqueeze(0).to(device)  # [1, seq_len, 768]


def predict_hate(model, text_embedding, image_embedding, p=0.5):
    """Predict if the image and text combination is hate speech."""
    with torch.no_grad():
        output = model(text_embedding, image_embedding)
        probabilities = F.softmax(output, dim=1)
    predicted_class = torch.argmax(output, dim=1).item()
    prob_class_0 = probabilities[0, 0].item()
    prob_class_1 = probabilities[0, 1].item()
    print(f"Probability of hate: {prob_class_1:.4f}, non-hate: {prob_class_0:.4f}")
    # predicted_class = torch.argmax(output, dim=1).item()
    predicted_class = 1 if prob_class_1 >= p else 0

    return predicted_class


def process_frames(frames_path, hate_classifier_path, device, p):
    """ "Process the frames one by one and predict if they contain hate. The prediction is done using the frame image and the text extracted from the frame.
    Returns a list of hateful frames to blur in the final video."""
    to_blur = []
    frames = [f for f in os.listdir(frames_path)]
    frames = sorted(frames, key=extract_frame_number)

    print("Loading models...")
    (
        feature_extractor,
        model_img,
        tokenizer_tweet,
        model_tweet,
        tokenizer_ocr,
        model_ocr,
        hate_classifier,
    ) = load_all_models(device, hate_classifier_path)
    total_frames = len(frames)
    i = 0
    for frame in frames:
        i += 1
        print("\nProcessing frame :", i, "/", total_frames, "=>", frame)
        image_path = os.path.join(frames_path, frame)

        image_embedding = get_image_embedding(
            image_path, feature_extractor, model_img, device
        )
        print("Frame image encoded.")

        text_embedding = get_text_embedding(
            image_path, tokenizer_ocr, model_ocr, tokenizer_tweet, model_tweet, device
        )
        print("OCR text encoded.")

        predicted_class = predict_hate(
            hate_classifier, text_embedding, image_embedding, p
        )
        print("Predicted class:", predicted_class)

        if predicted_class == 1:
            to_blur.append(frame)

    print("\nFrames to blur:", to_blur)
    return to_blur


def extract_frame_numbers(to_blur_paths):
    """Extract frame numbers from the list of paths."""
    frame_numbers = []
    for path in to_blur_paths:
        numbers = re.findall(r"\d+", path)
        if numbers:
            frame_numbers.append(int(numbers[-1]))  # Take last number if multiple
    return frame_numbers


def blur_video_frames(video_path, to_blur, output_video_path):
    """Blurs the specified frames in a video and saves the output."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    to_blur_numbers = extend_blur_numbers(extract_frame_numbers(to_blur))

    success = True
    frame_idx = 0

    while success:
        success, frame = cap.read()
        if not success:
            break
        if any(number - 29 <= frame_idx <= number + 29 for number in to_blur_numbers):
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            blurred_pil = pil_frame.filter(ImageFilter.GaussianBlur(radius=20))
            frame = cv2.cvtColor(np.array(blurred_pil), cv2.COLOR_RGB2BGR)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"censored frames video saved at {output_video_path}")


def extend_blur_numbers(to_blur_numbers, frequency=30):
    """Keep all original numbers and add extra ones if i-frequency and i+frequency exist."""
    if not to_blur_numbers:
        return []

    to_blur_set = set(to_blur_numbers)
    extra_numbers = []

    for i in range(max(to_blur_numbers) + 1):
        if (
            (i - 30 in to_blur_set)
            and (i + 30 in to_blur_set)
            and (i not in to_blur_set)
        ):
            extra_numbers.append(i)

    return sorted(to_blur_numbers + extra_numbers)


def update_audio(input_video, censored_audio, output_video):
    """Update the audio of the video with the censored audio."""
    temp_output = output_video + ".temp.mp4"

    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i",
        input_video,
        "-i",
        censored_audio,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        temp_output,
        "-y",
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Replace the original file
    shutil.move(temp_output, output_video)

    print("Censored video saved as:", output_video)


def clean_up(audio_path, frames_path, output_audio):
    """Delete audio, frames, and output audio directories."""
    if os.path.exists(audio_path):
        shutil.rmtree(audio_path)
    if os.path.exists(frames_path):
        shutil.rmtree(frames_path)
    if os.path.exists(output_audio):
        shutil.rmtree(output_audio)

    print("Cleaned up temporary files.")


def wrap_text(text, width=60):
    return "\n".join(textwrap.wrap(text, width))


def classify_video(df, to_blur, beep_intervals):
    """
    Classifies the video as Hateful or Clean based on thresholds.

    Criteria:
        - At least 3 hateful sentences
        - At least 3 frames to blur

    Args:
        df (pd.DataFrame): DataFrame with sentence classification results.
        to_blur (list): List of frame indices or filenames marked for blurring.

    Returns:
        dict: Classification summary.
    """
    num_hateful_sentences = len(beep_intervals)
    num_blurred_frames = len(to_blur)

    high_confidence_hate = df["prob_class_1"] > 0.8
    num_high_confidence = high_confidence_hate.sum()

    video_is_hateful = (
        num_hateful_sentences >= 5
        or num_blurred_frames >= 15
        or num_high_confidence >= 3
    )

    print("\n=== Video Classification Summary ===")
    print("Number of hateful sentences:", num_hateful_sentences)
    print("Number of high confidence hate sentences:", num_high_confidence)
    print("Number of hateful frames:", num_blurred_frames)
    print("Video is hateful:", video_is_hateful)
    print("=====================================")

    return video_is_hateful
