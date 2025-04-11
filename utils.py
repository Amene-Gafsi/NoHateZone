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


def extract_frames(input_path, output_dir, frequency=30):
    """
    Extract frames from video file and save them to output_dir.
    Records frame number and timestamp in a DataFrame.

    Args:
        input_path (str): Path to the input video file.
        output_dir (str): Path to the output directory where frames will be saved.
        frequency (int): Frequency of frames to save (e.g., save every nth frame).

    Returns:
        tuple: (number of frames extracted, DataFrame with 'frame_number' and 'timestamp' columns)
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
            filename = os.path.join(output_dir, f"frame{count}.jpg")
            cv2.imwrite(filename, image)
            records.append((count, timestamp))
            frame_count += 1

        count += 1

    cap.release()

    df = pd.DataFrame(records, columns=["frame_number", "timestamp_seconds"])
    return frame_count, df
