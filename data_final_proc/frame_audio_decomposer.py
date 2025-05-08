from audio2text import AudioTranscriber  # 16sec
from utils import extract_audio, extract_frames, create_chunk_dataframe, combine_text
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import tempfile
import re
import zipfile



VIDEO_SUFFIXES = {'.mp4', '.avi', '.mkv'}

def idx_from_name(name: str) -> int:
    m = re.search(r'_(\d+)$', name)
    return int(m.group(1)) if m else 0

def list_all_videos(video_zip_dir: Path):
    entries = []
    for zip_path in sorted(video_zip_dir.glob('*.zip'), key=lambda p: idx_from_name(p.stem)):
        with zipfile.ZipFile(zip_path, 'r') as z:
            for member in z.namelist():
                if Path(member).suffix.lower() in VIDEO_SUFFIXES:
                    entries.append((zip_path, member, Path(member).stem))
    entries.sort(key=lambda e: (0 if e[2].startswith('hate_video_') else 1,
                                idx_from_name(e[2])))
    return entries

def process_video_entry(
    zip_path: Path,
    member: str,
    vid_name: str,
    frames_root: Path,
    audio_root: Path,
    transcriber: AudioTranscriber,
    frequency: int = 30
):
    # temporary workspace
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1) unzip
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extract(member, tmpdir)
        video_file = Path(tmpdir) / member

        # 2) prepare outputs
        out_frames = frames_root / vid_name
        out_audio  = audio_root  / vid_name
        out_frames.mkdir(parents=True, exist_ok=True)
        out_audio.mkdir(parents=True, exist_ok=True)

        # 3) extract frames
        from utils import extract_frames, extract_audio  # imported here to avoid circular
        _, frame_data = extract_frames(
            str(video_file), str(out_frames),
            video_index=idx_from_name(vid_name), frequency=frequency
        )
        frame_data['video'] = frame_data['frame_number'].apply(
            lambda f: f"{vid_name}_frame{f}.jpg"
        )
        frame_df = frame_data[['timestamp_seconds', 'video', 'frame_number']]

        # 4) extract audio & transcribe
        chunk_df = pd.DataFrame(columns=['text', 'start', 'end', 'video'])
        try:
            extract_audio(input_path=str(video_file), output_dir=str(out_audio))
            audio_file = out_audio / f"{vid_name}.mp3"

            raw = transcriber.transcribe_audio(str(audio_file))
            sentences = transcriber.extract_sentences_with_timestamps(raw)

            if sentences:
                chunk_df = pd.DataFrame(sentences)
                chunk_df['video'] = vid_name

        except Exception as e:
            print(f"⚠️ Audio processing failed for {vid_name}: {e}")

    return frame_df, chunk_df

if __name__ == '__main__':
    cwd = Path.cwd()
    video_zip_dir = cwd / 'input' / 'video_trial'
    frames_root   = cwd / 'input' / 'frames'
    audio_root    = cwd / 'input' / 'audio'

    frames_root.mkdir(parents=True, exist_ok=True)
    audio_root.mkdir(parents=True, exist_ok=True)

    transcriber = AudioTranscriber(chunk_length=30)
    video_entries = list_all_videos(video_zip_dir)

    all_frame_dfs = []
    all_chunk_dfs = []
    for zip_path, member, vid_name in tqdm(video_entries, desc='Processing videos'):
        fdf, cdf = process_video_entry(
            zip_path, member, vid_name,
            frames_root, audio_root, transcriber
        )
        all_frame_dfs.append(fdf)
        all_chunk_dfs.append(cdf)

    final_frames_df = pd.concat(all_frame_dfs, ignore_index=True)
    final_chunks_df = pd.concat(all_chunk_dfs, ignore_index=True)
    final_frames_df.to_csv('frames.csv', index=False)
    final_chunks_df.to_csv('chunks.csv', index=False)

    # annotate frames per sentence
    frames = final_frames_df.copy()
    frames['video_id'] = frames['video'].str.replace(r'_frame\d+\.jpg$', '', regex=True)

    def find_frames(row):
        vid, s, e = row['video'], row['start'], row['end']
        mask = (
            (frames['video_id'] == vid) &
            (frames['timestamp_seconds'] >= s) &
            (frames['timestamp_seconds'] <= e)
        )
        return frames.loc[mask, 'video'].tolist()

    annotated = final_chunks_df.copy()
    annotated['frames_within_interval'] = annotated.apply(find_frames, axis=1)
    annotated.to_csv('annotated_chunks.csv', index=False)
