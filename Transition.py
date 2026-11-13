

# 1. Import libraries
import cv2
import numpy as np
import sys
from pathlib import Path


# 2. Define project paths
BASE_DIR = Path(__file__).resolve().parent

FILM_REPO_DIR = BASE_DIR / "frame-interpolation"
sys.path.insert(0, str(FILM_REPO_DIR))

from eval import interpolator as interpolator_lib
from eval import util


INPUT_DIR = BASE_DIR / "input_videos"
OUTPUT_DIR = BASE_DIR / "transition_input"

MODEL_PATH = (
    BASE_DIR
    / "pretrained_models"
    / "film_net"
    / "Style"
    / "saved_model"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 3. Extract frames from videos
def extract_frame(video_path, frame_position, output_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < 1:
        cap.release()
        raise RuntimeError("No frames found")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)

    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError("Could not extract frame")

    cv2.imwrite(str(output_path), frame)

    print(f"Frame saved: {output_path}")

    return total_frames


# 4. Generate 15 intermediate frames using FILM
def run_film(frame1_path, frame2_path):

    print("\nStarting FILM interpolation...")

    interpolator = interpolator_lib.Interpolator(
        model_path=str(MODEL_PATH),
        align=64,
        block_shape=[1, 1]
    )

    image_1 = util.read_image(str(frame1_path))
    image_2 = util.read_image(str(frame2_path))

    image_batch_1 = np.expand_dims(image_1, axis=0)
    image_batch_2 = np.expand_dims(image_2, axis=0)

    # Generate 15 intermediate frames
    time_values = [i / 16 for i in range(1, 16)]

    for t in time_values:

        batch_dt = np.full(
            shape=(1,),
            fill_value=t,
            dtype=np.float32
        )

        mid_frame = interpolator(
            image_batch_1,
            image_batch_2,
            batch_dt
        )[0]

        output_path = OUTPUT_DIR / f"film_{int(t * 1000):03d}.png"

        util.write_image(str(output_path), mid_frame)

        print(f"Generated frame at t={t}: {output_path}")

    print("\nAll 15 intermediate frames generated!")


# 5. Merge both videos with the FILM transition
def merge_videos_with_transition():

    print("\nStarting video merging...")

    video1_path = INPUT_DIR / "video1.mp4"
    video2_path = INPUT_DIR / "video2.mp4"

    # Read video properties
    cap = cv2.VideoCapture(str(video1_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open video1.mp4")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    # Create output video
    output_path = BASE_DIR / "merged_video.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError("Could not create output video")

    try:

        # STEP 1: Add all frames from Video 1
        print("Adding Video 1...")

        cap = cv2.VideoCapture(str(video1_path))

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            writer.write(frame)

        cap.release()

        # STEP 2: Add FILM-generated frames
        print("Adding FILM transition frames...")

        transition_frames = sorted(
            OUTPUT_DIR.glob("film_[0-9][0-9][0-9].png"),
            key=lambda p: int(p.stem.split("_")[1])
        )

        if len(transition_frames) != 15:
            raise RuntimeError(
                f"Expected 15 frames, found {len(transition_frames)}"
            )

        for image_path in transition_frames:

            frame = cv2.imread(str(image_path))

            if frame is None:
                raise RuntimeError(f"Could not read {image_path}")

            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            writer.write(frame)

        # STEP 3: Add all frames from Video 2
        print("Adding Video 2...")

        cap = cv2.VideoCapture(str(video2_path))

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            writer.write(frame)

        cap.release()

    finally:
        writer.release()

    print("\nVideo merging completed!")
    print(f"Final video: {output_path}")


# 6. Main function
def main():

    video1_path = INPUT_DIR / "video1.mp4"
    video2_path = INPUT_DIR / "video2.mp4"

    frame1_path = OUTPUT_DIR / "frame1.png"
    frame2_path = OUTPUT_DIR / "frame2.png"

    if not video1_path.exists():
        raise FileNotFoundError(video1_path)

    if not video2_path.exists():
        raise FileNotFoundError(video2_path)

    print("Extracting frames...")

    # Extract last frame from Video 1
    cap = cv2.VideoCapture(str(video1_path))

    if not cap.isOpened():
        raise RuntimeError("Cannot open Video 1")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    extract_frame(
        video1_path,
        total_frames - 1,
        frame1_path
    )

    # Extract first frame from Video 2
    extract_frame(
        video2_path,
        0,
        frame2_path
    )

    print("\nBoth frames extracted successfully!")

    # Generate intermediate frames
    run_film(
        frame1_path,
        frame2_path
    )

    # Merge original videos and transition frames
    merge_videos_with_transition()

    print("\nAll operations completed!")


# 7. Execute program
if __name__ == "__main__":
    main()
