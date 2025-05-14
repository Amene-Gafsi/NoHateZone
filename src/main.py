import argparse
import os
import warnings

import torch
from transformers.utils import logging
import pandas as pd

from utils import *

# Remove warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()


def main(media_dir, checkpoints_dir):
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    finetuned_distilbert_path = os.path.join(checkpoints_dir, "distilbert_hatespeech")
    fusion_model_path = os.path.join(checkpoints_dir, "model_epoch_18.pt")

    video_path = os.path.join(media_dir, "input", "video", "video.mp4")
    audio_path = os.path.join(media_dir, "input", "audio")
    frames_path = os.path.join(media_dir, "input", "frames")
    output_audio = os.path.join(media_dir, "output", "audio")
    output_audio_path = os.path.join(output_audio, "censored_audio.wav")
    output_video_path = os.path.join(media_dir, "output", "video", "censored_video.mp4")

    print("\nExtracting audio and frames...")
    input_audio = extract_audio_and_frames(video_path, audio_path, frames_path)

    print("\nTranscribing audio...")
    transcription = transcribe_audio(input_audio, device)

    print("Transcription :", transcription["text"])

    print("\nClassifying sentences...")
    df = classify_sentences(transcription["text"], finetuned_distilbert_path)
    print(df)

    print("\nFinding beep time intervals...")
    beep_intervals = find_beep_intervals(df, transcription)

    print("\nGenerating censored audio...")
    censor_audio(input_audio, beep_intervals, output_audio_path)

    print("\nProcessing frames...")
    to_blur = process_frames(frames_path, fusion_model_path, device, p=0.79)

    print("\nGenerating censored video...")
    blur_video_frames(
        video_path=video_path,
        to_blur=to_blur,
        output_video_path=output_video_path,
    )

    print("\nUpdating video with censored audio...")
    update_audio(output_video_path, output_audio_path, output_video_path)

    print("\nCleaning up...")
    clean_up(audio_path, frames_path, output_audio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Censor hate speech from videos.")
    parser.add_argument(
        "--media_dir",
        type=str,
        required=True,
        help="Path to the media directory (containing input/output folders)",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        required=True,
        help="Path to the directory containing model checkpoints",
    )
    args = parser.parse_args()
    main(args.media_dir, args.checkpoints_dir)
    # main("../media", "../checkpoints")
