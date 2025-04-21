from transformers import AutoTokenizer, AutoModel, logging
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import re
import torch
import gc

# ─── CONFIG ────────────────────────────────────────────────────────────────
logging.set_verbosity_error()

cwd        = Path.cwd()
local_dir  = cwd / 'OCR_impl_new' / "got_ocr2_0"   # path to your GOT‑OCR2 model folder
frames_dir = cwd / 'OCR_impl_new' / "frames"       # directory containing video_* folders
output_csv = cwd / 'OCR_impl_new' / "ocr_results.csv"  # where to save the CSV

# ─── LOAD MODEL ────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(local_dir, 
                                          trust_remote_code=True,
                                          local_files_only = True)
model     = AutoModel.from_pretrained(
    local_dir,
    trust_remote_code=True,
    use_safetensors=True,
    device_map="auto",
    pad_token_id=tokenizer.eos_token_id,
    local_files_only = True
)
model.config.pad_token_id = tokenizer.eos_token_id
model.eval()

def clear_cache():
    """Free up GPU & Python memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def frame_index(path: Path):
    """Extract trailing integer from filename, or return -1."""
    m = re.search(r"(\d+)(?=\.jpg$)", path.name)
    return int(m.group(1)) if m else -1

# ─── MAIN OCR LOOP ─────────────────────────────────────────────────────────
records = []
if not frames_dir.exists():
    raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

# find all subfolders named video_* (e.g. hate_video_1, non_hate_video_2)
video_dirs = sorted(
    [d for d in frames_dir.iterdir() if d.is_dir() and re.match(r"(?:non_)?hate_video_\d+$", d.name)],
    key=lambda d: int(re.search(r"(\d+)$", d.name).group(1))
)
print(f"Found {len(video_dirs)} video directories: {[d.name for d in video_dirs]}")

with torch.no_grad():
    for video_dir in tqdm(video_dirs, desc="Videos", unit="video"):
        clear_cache()
        # gather all .jpg frames in sorted order
        frames = sorted(video_dir.glob("*.jpg"), key=frame_index)
        for frame_path in tqdm(frames, desc=f"Frames in {video_dir.name}", unit="frame", leave=False):
            clear_cache()
            try:
                # perform OCR
                result = model.chat(
                    tokenizer,
                    str(frame_path),
                    ocr_type="ocr"
                )
            except Exception as e:
                print(f"[WARN] {video_dir.name}/{frame_path.name} failed: {e}")
                result = ""
            # record the output
            records.append({
                "video": video_dir.name,
                "frame": frame_path.name,
                "text":  result.strip()
            })

# ─── SAVE RESULTS ──────────────────────────────────────────────────────────
df = pd.DataFrame(records, columns=["video", "frame", "text"])
df.to_csv(output_csv, index=False)
print(f"\n✓ Finished! Wrote {len(df)} rows to {output_csv}")
