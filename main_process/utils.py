import cv2
import os
import pandas as pd
import audio_extract
import pytesseract



def extract_audio(input_path, output_dir):
    """Extract audio from video file defined by input_path and save it to output_path.
    Args:
        input_path (str): Path to the input video file.
        output_dir (str): Path to the output directory where audio will be saved.
    """

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, base_name + ".mp3")

    if os.path.exists(output_path):
        os.remove(output_path)

    return audio_extract.extract_audio(input_path, output_path)



def extract_frames(input_path, output_dir, video_index, frequency=30):
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
            filename = os.path.join(output_dir, f"video_{video_index}_frame{count}.jpg")
            cv2.imwrite(filename, image)
            records.append((count, timestamp))
            frame_count += 1

        count += 1

    cap.release()

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
