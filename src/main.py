import argparse
import os
import warnings

import pandas as pd
import torch
from transformers.utils import logging

from utils.utils import *

warnings.filterwarnings("ignore")
logging.set_verbosity_error()


def main(media_dir, checkpoints_dir):
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    finetuned_distilbert_path = os.path.join(checkpoints_dir, "distilbert_hatespeech")
    fusion_model_path = os.path.join(checkpoints_dir, "finetuned_HMM/model_hatemm.pt")

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

    print("\nFinding beep time intervals...")
    beep_intervals = find_beep_intervals(df, transcription)

    print("\nGenerating censored audio...")
    censor_audio(input_audio, beep_intervals, output_audio_path)

    print("\nProcessing frames...")
    # adjsut the classification threshold p depending on how conservative you want the censoring to be
    to_blur = process_frames(frames_path, fusion_model_path, device, p=0.5)

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

    # We also provilde the option to classify the entire video based on the frames-audio results
    # classify_video(df, to_blur, beep_intervals)


if __name__ == "__main__":
    # specify the media and checkpoints directories
    # default values are set to "../media" and "../checkpoints"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_media_dir = os.path.abspath(os.path.join(current_dir, "..", "media"))
    default_checkpoints_dir = os.path.abspath(
        os.path.join(current_dir, "..", "checkpoints")
    )

    parser = argparse.ArgumentParser(description="Censor hate speech from videos.")
    parser.add_argument(
        "--media_dir",
        type=str,
        # required=True,
        default=default_media_dir,
        help="Path to the media directory (containing input/output folders)",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        # required=True,
        default=default_checkpoints_dir,
        help="Path to the directory containing model checkpoints",
    )
    args = parser.parse_args()
    main(args.media_dir, args.checkpoints_dir)
