import os
import re
import sys
from pathlib import Path

import cv2
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))

from audio2text import AudioTranscriber

from utils import extract_audio


def parse_timestamp(ts: str) -> float:
    """
    Convert 'HH:MM:SS' or 'MM:SS' into seconds.
    """
    parts = [float(p) for p in ts.split(":")]
    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    return parts[0]


def get_video_duration(video_path: Path) -> float:
    """
    Return video duration in seconds using OpenCV.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return (frames / fps) if fps > 0 and frames > 0 else 0.0


def safe_get_text_in_interval(asr_result: dict, start: float, end: float) -> str:
    """
    Concatenate all ASR word-chunks wholly within [start,end].
    """
    out = []
    for chunk in asr_result.get("chunks", []):
        s, e = chunk.get("timestamp", (None, None))
        if s is None or e is None:
            continue
        if s >= start and e <= end:
            out.append(chunk["text"].strip())
    return " ".join(out)


def build_labeled_transcripts(
    annotations: pd.DataFrame,
    hate_dir: str,
    non_hate_dir: str,
    audio_root: str,
    output_csv: str,
) -> pd.DataFrame:
    """
    For each row in annotations:
      - Extract audio, run word-level ASR
      - If video is a hate_video:
          * Parse its annotated snippet intervals
          * For each snippet: label = Audio_hate (0 or 1)
          * Build complementary intervals (before/after), always label = 0
      - If video is non_hate_video:
          * Single interval [0, duration], label = 0
    Dump all intervals to output_csv.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transcriber = AudioTranscriber(chunk_length=30, device=device)
    records = []

    for _, row in tqdm(annotations.iterrows(), total=len(annotations), desc="Videos"):
        fn = row["video_file_name"]
        stem = Path(fn).stem
        is_hate_file = stem.startswith("hate_video")
        snippet_label = int(row.get("Audio_hate", 0))

        # locate the correct folder
        folder = hate_dir if is_hate_file else non_hate_dir
        video_path = Path(folder) / fn
        if not video_path.exists():
            print(f"⚠️ Missing video {video_path}")
            continue

        # get true duration
        duration = get_video_duration(video_path)
        if duration <= 0:
            print(f"⚠️ Invalid duration for {fn}")
            continue

        # extract audio
        out_dir = Path(audio_root) / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            extract_audio(str(video_path), str(out_dir))
        except Exception as e:
            print(f"⚠️ Audio extract failed {fn}: {e}")
            continue
        mp3 = out_dir / f"audio.mp3"

        # word-level ASR
        try:
            asr_res = transcriber.transcribe_audio(str(mp3), return_timestamps="word")
        except Exception as e:
            print(f"⚠️ ASR failed {fn}: {e}")
            continue

        segments = []
        if is_hate_file:
            # parse annotation
            raw = row.get("hate_snippet", [])
            if isinstance(raw, str):
                try:
                    raw = eval(raw)
                except:
                    raw = []
            intervals = []
            for item in raw:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    print(f"⚠️ Skipping malformed snippet in {fn}: {item}")
                    continue
                start_ts, end_ts = item
                try:
                    s = max(0.0, min(parse_timestamp(start_ts), duration))
                    e = max(0.0, min(parse_timestamp(end_ts), duration))
                except Exception as e:
                    print(f"⚠️ Timestamp parsing failed in {fn}: {e}")
                    continue
                if s < e:
                    intervals.append((s, e))

            intervals.sort()

            # build [before first],[snippet],[between],…,[after last]
            cur = 0.0
            for s, e in intervals:
                if cur < s:
                    # non-hate interval
                    segments.append((cur, s, 0))
                # annotated interval with its actual label
                segments.append((s, e, snippet_label))
                cur = e
            if cur < duration:
                segments.append((cur, duration, 0))
        else:
            # single non-hate segment
            segments = [(0.0, duration, 0)]

        # extract texts
        for s, e, lab in segments:
            text = safe_get_text_in_interval(asr_res, s, e)
            records.append(
                {
                    "video_file_name": fn,
                    "start": s,
                    "end": e,
                    "text": text,
                    "label": lab,
                }
            )

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df)} rows to {output_csv}")
    return df


if __name__ == "__main__":
    # HATE_DIR = "try_hate"
    # NON_HATE_DIR = "try_non_hate"
    # -----------------paths to hate and non-hate videos-----------------
    # HATE_DIR = "/home/gafsi/NoHateZone/data/HateMM/HateMM/hate_videos/hate_videos"
    root_path = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(root_path, "../..")
    HATE_DIR = os.path.join(root_dir, "data/HateMM/HateMM/hate_videos/hate_videos")
    # NON_HATE_DIR = (
    #     "/home/gafsi/NoHateZone/data/HateMM/HateMM/non_hate_videos/non_hate_videos"
    # )
    NON_HATE_DIR = os.path.join(
        root_dir, "data/HateMM/HateMM/non_hate_videos/non_hate_videos"
    )
    # -------------------------------------------------------------------
    AUDIO_ROOT = "audio_extracted"
    # ANNO_XLSX = "/home/gafsi/NoHateZone/src/hatemm_audio_ts_labels.xlsx"
    ANNO_XLSX = os.path.join(root_dir, "data/HateMM/hatemm_audio_ts_labels.xlsx")
    OUTPUT_CSV = "hatemm_audio_testset.csv"

    # test-mode: pick first 3 of each
    TEST_MODE = False
    TEST_HATE = 3
    TEST_NON = 3

    ann = pd.read_excel(ANNO_XLSX)
    if TEST_MODE:
        hates = ann[ann["video_file_name"].str.startswith("hate_video", na=False)].head(
            TEST_HATE
        )
        nons = ann[
            ann["video_file_name"].str.startswith("non_hate_video", na=False)
        ].head(TEST_NON)
        ann = pd.concat([hates, nons], ignore_index=True)
        print(f"TEST MODE: {len(hates)} hate + {len(nons)} non-hate videos")

    print("Hate count:", ann["video_file_name"].str.startswith("hate_video").sum())
    print(
        "Non-hate count:", ann["video_file_name"].str.startswith("non_hate_video").sum()
    )

    df = build_labeled_transcripts(
        ann,
        hate_dir=HATE_DIR,
        non_hate_dir=NON_HATE_DIR,
        audio_root=AUDIO_ROOT,
        output_csv=OUTPUT_CSV,
    )

    if TEST_MODE:
        print("\n=== Test segments ===")
        print(df)
