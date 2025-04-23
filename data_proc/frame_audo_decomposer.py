from audio2text import AudioTranscriber  # 16sec
from utils import extract_audio, extract_frames, create_chunk_dataframe, combine_text
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import tempfile
import re
import zipfile

# Initialize transcriber
transcriber = AudioTranscriber(chunk_length=30)
VIDEO_SUFFIXES = {'.mp4', '.avi', '.mkv'}

def idx_from_name(name: str) -> int:
    """Extract trailing integer from names like 'hate_video_12' or 'non_hate_video_3'."""
    m = re.search(r'_(\d+)$', name)
    return int(m.group(1)) if m else 0

def list_all_videos(video_zip_dir: Path):
    """
    For each .zip in the directory, collect (zip_path, member_name, vid_name)
    for every video file inside, sorted hate_ before non_hate_, then numerically.
    """
    entries = []
    for zip_path in sorted(video_zip_dir.glob('*.zip'), key=lambda p: idx_from_name(p.stem)):
        with zipfile.ZipFile(zip_path, 'r') as z:
            for member in z.namelist():
                if Path(member).suffix.lower() in VIDEO_SUFFIXES:
                    vid_name = Path(member).stem
                    entries.append((zip_path, member, vid_name))
    entries.sort(key=lambda e: (
        0 if e[2].startswith('hate_video_') else 1,
        idx_from_name(e[2])
    ))
    return entries

def process_video_entry(zip_path: Path, member: str, vid_name: str,
                        frames_root: Path, audio_root: Path,
                        transcriber, frequency: int = 30):
    """
    Unzip the specified member, extract frames & audio into
    frames_root/vid_name/ and audio_root/vid_name/, transcribe, and return two DataFrames.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1) extract this single file
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extract(member, tmpdir)
        video_file = Path(tmpdir) / member

        # 2) prepare output folders
        out_frames = frames_root / vid_name
        out_frames.mkdir(parents=True, exist_ok=True)
        out_audio  = audio_root  / vid_name
        out_audio.mkdir(parents=True, exist_ok=True)

        # 3) extract frames
        _, frame_data = extract_frames(
            str(video_file), str(out_frames),
            video_index=idx_from_name(vid_name),
            frequency=frequency
        )
        frame_data['video'] = frame_data['frame_number'] \
            .apply(lambda f: f"{vid_name}_frame{f}.jpg")
        frame_df = frame_data[['timestamp_seconds', 'video', 'frame_number']]

        # 4 & 5) extract audio and transcribe with unified fallback
        chunk_df = pd.DataFrame(columns=["timestamp", "text", "video"])
        try:
            # extract audio
            extract_audio(input_path=str(video_file), output_dir=str(out_audio))
            audio_file = out_audio / f"{vid_name}.mp3"
            # transcribe audio
            result     = transcriber.transcribe_audio(
                              input_audio=str(audio_file),
                              return_timestamps=True
                          )
            chunk_info = transcriber.extract_segments_with_timestamps(result)
            chunk_df   = create_chunk_dataframe(chunk_info)
            chunk_df['video'] = vid_name
        except Exception as e:
            print(f"⚠️ Audio processing failed for {vid_name}: {e}")
            # chunk_df remains empty

    return frame_df, chunk_df

# ─── MAIN ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cwd           = Path.cwd()
    video_zip_dir = cwd /'data_proc'/ 'input' / 'video'
    frames_root   = cwd / 'data_proc'/'input' / 'frames'
    audio_root    = cwd / 'data_proc'/ 'input' / 'audio'

    # ensure output dirs exist
    frames_root.mkdir(parents=True, exist_ok=True)
    audio_root.mkdir(parents=True, exist_ok=True)

    # build the list of all videos inside all zips
    video_entries = list_all_videos(video_zip_dir)
    # optionally limit number of videos:
    #video_entries = video_entries[:542]

    all_frame_dfs = []
    all_chunk_dfs = []

    # process them sequentially
    for zip_path, member, vid_name in tqdm(video_entries, desc="Processing videos", unit="video"):
        fdf, cdf = process_video_entry(
            zip_path, member, vid_name,
            frames_root, audio_root,
            transcriber
        )
        all_frame_dfs.append(fdf)
        all_chunk_dfs.append(cdf)

    # combine and save results
    final_frames_df = pd.concat(all_frame_dfs, ignore_index=True)
    final_chunks_df = pd.concat(all_chunk_dfs, ignore_index=True)
    # write CSVs
    final_frames_df.to_csv('frames.csv', index=False)
    final_chunks_df.to_csv('chunks.csv', index=False)

    # annotation
    def annotate_frames_within_interval(frames_df: pd.DataFrame, chunks_df: pd.DataFrame) -> pd.DataFrame:
        out = chunks_df.copy()
        out[['start', 'end']] = pd.DataFrame(
            out['timestamp']
                .apply(lambda ts: (float(ts[0]), float(ts[1])))
                .tolist(),
            index=out.index
        )
        if 'video_id' not in frames_df:
            frames_df = frames_df.copy()
            frames_df['video_id'] = frames_df['video'].str.replace(
                r'_frame\d+\.jpg$', '', regex=True
            )
        def find_frames(row):
            vid, s, e = row['video'], row['start'], row['end']
            mask = (
                (frames_df['video_id'] == vid) &
                (frames_df['timestamp_seconds'] >= s) &
                (frames_df['timestamp_seconds'] <= e)
            )
            return frames_df.loc[mask, 'video'].tolist()
        out['frames_within_interval'] = out.apply(find_frames, axis=1)
        return out.drop(columns=['start', 'end'])

    annotated_chunks = annotate_frames_within_interval(final_frames_df, final_chunks_df)
    annotated_chunks.to_csv('annotated_chunks.csv', index=False)
