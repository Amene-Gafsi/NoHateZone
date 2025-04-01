import cv2
import os
import audio_extract


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
    Extract frames from video file defined by input_path and save it to output_dir.
    Args:
        input_path (str): Path to the input video file.
        output_dir (str): Path to the output directory where frames will be saved.
        frequency (int): Frequency of frames to save (e.g., save every nth frame).
    Returns:
        Int: Number of frames extracted.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    count = 0
    frame_count = 0
    success = True

    while success:
        success, image = cap.read()
        if not success:
            break

        if count % frequency == 0:
            cv2.imwrite(os.path.join(output_dir, f"frame{count}.jpg"), image)
            frame_count += 1

        count += 1

    cap.release()
    return frame_count
