from audio2text import AudioTranscriber  # 16sec
from utils import extract_audio, extract_frames, create_chunk_dataframe, combine_text
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import tempfile
import re
import zipfile

final_annot = pd.read_excel("HateMM_annotation_final.xlsx")

audio_data = final_annot[["video_file_name","hate_snippet","Audio_hate"]]

# extract the numeric index from any "hate_video_#.mp4" filenames
nums = (
    audio_data['video_file_name']
    .str
    .extract(r'hate_video_(\d+)\.mp4')[0]      # grab the digits
    .astype(float, errors='ignore')             # convert to number; non-matches become NaN
)

# now select only those rows where the index is ≤ 181
filtered = audio_data[nums <= 181]

# filter the non-hate videos and fill their Audio_hate label as 1
filtered['Audio_hate'] = filtered['Audio_hate'].fillna(0)

# boolean mask for hate videos
is_hate = filtered['video_file_name'].str.startswith('hate_video')

# counts
n_hate     = is_hate.sum()
n_non_hate = (~is_hate).sum()

print(f"Hate videos: {n_hate}")
print(f"Non-hate videos: {n_non_hate}")

filtered.to_excel("filtered_hateMM_annotation_final.xlsx", index=False)



