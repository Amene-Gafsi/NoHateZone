#!/usr/bin/env python3
import ast
import math
import subprocess
import sys
import zipfile
import tempfile
from glob import glob
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ─── CONFIG ────────────────────────────────────────────────────────────────
cwd = Path.cwd()
FRAMES_CSV = cwd / 'results_org' / 'frames_final_org.csv'
CHUNKS_CSV = cwd / 'results_org' / 'chunks_final_org.csv'
AUDIO_ROOT = cwd / 'input' / 'audio'
VIDEO_ROOT = cwd / 'input' / 'video'
VIDEO_EXT = '.mp4'
OUTPUT_DIR = cwd / 'output'
TRANSCRIPT_CHUNK_LENGTH = 30

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── HELPERS ───────────────────────────────────────────────────────────────

def ffprobe_duration(path: Path) -> float | None:
    """Return duration in seconds using ffprobe, or None if ffprobe fails."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(path)
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return float(output.strip())
    except Exception:
        return None


def parse_interval(raw) -> tuple[float, float]:
    """Convert a string like '[x, y]' to (start, end) floats, or (nan, nan)."""
    try:
        start, end = ast.literal_eval(str(raw))
        return float(start), float(end)
    except Exception:
        return math.nan, math.nan


def validate_intervals(df: pd.DataFrame,
                       start_col: str = 'start',
                       end_col: str = 'end',
                       video_col: str = 'video') -> pd.DataFrame:
    """
    Return DataFrame of rows where end <= start or start/end is NaN.
    """
    issues = []
    for idx, row in df.iterrows():
        s, e = row[start_col], row[end_col]
        if math.isnan(s) or math.isnan(e) or e <= s:
            issues.append({'row': idx, 'video': row[video_col], 'start': s, 'end': e})
    return pd.DataFrame(issues)


def select_longest_overlaps(df: pd.DataFrame,
                            start_col: str = 'start',
                            end_col: str = 'end',
                            video_col: str = 'video') -> pd.DataFrame:
    """
    For strict overlaps within each video, keep only the interval
    with the longest duration in each overlapping cluster.
    """
    keep_indices = []
    for vid, group in df.groupby(video_col):
        grp_sorted = group.sort_values(start_col)
        cluster, max_end = [], None

        def flush_cluster():
            if not cluster:
                return
            durations = [
                (i, grp_sorted.loc[i, end_col] - grp_sorted.loc[i, start_col])
                for i in cluster
            ]
            best_idx, _ = max(durations, key=lambda x: x[1])
            keep_indices.append(best_idx)

        for idx, row in grp_sorted.iterrows():
            s, e = row[start_col], row[end_col]
            if cluster and s < max_end:
                cluster.append(idx)
                max_end = max(max_end, e)
            else:
                flush_cluster()
                cluster = [idx]
                max_end = e
        flush_cluster()

    return df.loc[keep_indices].sort_index()


def collect_video_durations(root: Path, ext: str) -> pd.DataFrame:
    """Probe durations for loose and zipped videos, return DataFrame."""
    records = []
    # Loose files
    for p in root.rglob(f'*{ext}'):
        dur = ffprobe_duration(p)
        if dur is not None:
            records.append((p.relative_to(root).as_posix(), dur))
    # ZIP archives
    for z in root.rglob('*.zip'):
        with zipfile.ZipFile(z) as zf, tempfile.TemporaryDirectory() as tmpdir:
            for member in zf.namelist():
                if member.lower().endswith(ext):
                    extracted = Path(zf.extract(member, path=tmpdir))
                    dur = ffprobe_duration(extracted)
                    if dur is not None:
                        records.append((f'{z.name}/{member}', dur))
    return pd.DataFrame(records, columns=['video', 'duration_seconds'])


def transcribe_first_speech(audio_root: Path, chunk_length: int) -> pd.DataFrame:
    """
    Transcribe audio files and record timestamp of first spoken word.
    Falls back gracefully if timestamp decoding fails.
    """
    from audio2text import AudioTranscriber
    transcriber = AudioTranscriber(chunk_length=chunk_length)
    results = []

    for audio_path in tqdm(glob(str(audio_root / '**/*.mp3'),
                                recursive=True),
                           desc='Transcribing audio'):
        video_key = Path(audio_path).stem
        first_ts = None

        try:
            res = transcriber.transcribe_audio(
                input_audio=audio_path,
                return_timestamps='word'
            )
            chunks = res.get('chunks', [])
            for chunk in chunks:
                ts = chunk.get('timestamp')
                # ensure ts is a sequence and first element is not None
                if isinstance(ts, (list, tuple)) and ts and ts[0] is not None:
                    first_ts = float(ts[0])
                    break

        except Exception as e:
            print(f"Warning: could not get word‐level timestamps for {audio_path}: {e}",
                  file=sys.stderr)
            # fallback: leave first_ts as None

        results.append((video_key, first_ts))

    return pd.DataFrame(results, columns=['video', 'first_speech_time'])


def main():
    # 1) Load & round frame timestamps
    frames_df = pd.read_csv(FRAMES_CSV)
    frames_df['timestamp_seconds'] = frames_df['timestamp_seconds'].round(3)

    # 2) Load & parse chunk intervals
    chunks_df = pd.read_csv(CHUNKS_CSV)
    starts, ends = zip(*(parse_interval(ts) for ts in chunks_df['timestamp']))
    chunks_df['start'], chunks_df['end'] = starts, ends
    chunks_df.drop(columns=['timestamp'], inplace=True)

    # 3) Validate intervals & save issues
    issues_df = validate_intervals(chunks_df)
    issues_df.to_csv(OUTPUT_DIR / 'interval_issues.csv', index=False)

    # 4) Fill missing 'end' from video durations
    durations_df = collect_video_durations(VIDEO_ROOT, VIDEO_EXT)
    key_pattern = r'((?:non_)?hate_video_\d+)'
    chunks_df['video_key'] = chunks_df['video'].str.extract(key_pattern, expand=False)
    durations_df['video_key'] = durations_df['video'].str.extract(key_pattern, expand=False)

    dur_map = (
        durations_df
        .dropna(subset=['video_key'])
        .drop_duplicates('video_key')
        .set_index('video_key')['duration_seconds']
    )
    mask_missing_end = chunks_df['end'].isna()
    chunks_df.loc[mask_missing_end, 'end'] = chunks_df.loc[mask_missing_end, 'video_key'].map(dur_map)
    chunks_df.drop(columns=['video_key'], inplace=True)
    chunks_df.to_excel(OUTPUT_DIR / 'chunks_filled.xlsx', index=False)

    # 5) Remove overlapping intervals
    filtered_df = select_longest_overlaps(chunks_df)
    filtered_df.to_excel(OUTPUT_DIR / 'final_chunks.xlsx', index=False)

    # 6) Transcribe & insert first speech times
    speech_df = transcribe_first_speech(AUDIO_ROOT, TRANSCRIPT_CHUNK_LENGTH)
    final_df = filtered_df.copy()
    first_indices = final_df.groupby('video', sort=False).head(1).index
    speech_map = speech_df.set_index('video')['first_speech_time'].to_dict()

    for idx in first_indices:
        vid = final_df.at[idx, 'video']
        if vid in speech_map and speech_map[vid] is not None:
            final_df.at[idx, 'start'] = speech_map[vid]

    final_df.to_excel(OUTPUT_DIR / 'processed_chunks_final.xlsx', index=False)
    print(f"✓ Processing complete. Check outputs in {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
