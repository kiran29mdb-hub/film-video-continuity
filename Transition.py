

import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "input_videos"
OUTPUT_DIR = BASE_DIR / "transition_input"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_frame(video_path, frame_position, output_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < 1:
        cap.release()
        raise RuntimeError(f"No frames found: {video_path}")

    # Move to the exact requested frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)

    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError(f"Could not extract: {video_path}")

    cv2.imwrite(str(output_path), frame)

    print(f"Saved: {output_path}")
    print(f"Frame position: {frame_position}")
    print(f"Total frames: {total_frames}")


# Extract the actual last frame of Video 1
video1_frames = int(
    cv2.VideoCapture(str(INPUT_DIR / "video1.mp4"))
    .get(cv2.CAP_PROP_FRAME_COUNT)
)

extract_frame(
    INPUT_DIR / "video1.mp4",
    video1_frames - 1,
    OUTPUT_DIR / "frame1.png"
)

# Extract the first frame of Video 2
extract_frame(
    INPUT_DIR / "video2.mp4",
    0,
    OUTPUT_DIR / "frame2.png"
)

print("\nBoundary frames extracted successfully!")
