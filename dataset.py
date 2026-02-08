import cv2
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO

# --- Configuration ---
VIDEO_DIR = "/home/imtiaz/Documents/GitHub/activity_det/dataset"  # Your folder: Dataset/Walking/videos/...
OUTPUT_DIR = "/home/imtiaz/Documents/GitHub/activity_det/pose_dataset"
OUTPUT_CSV = "action_data.csv"
DATA_DIR = "/home/imtiaz/Documents/GitHub/activity_det/dataset"
SEQUENCE_LENGTH = 30  # Every sample will be exactly 30 frames
MODEL_PATH = "yolo11n-pose.pt"  # 'n' for nano is fastest for extraction

model = YOLO(MODEL_PATH)


def extract_landmarks(video_path, target_frames=30):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < target_frames:
        # If video is too short, we'll repeat some frames
        indices = np.linspace(0, total_frames - 1, target_frames).astype(int)
    else:
        # If video is long, we'll skip frames evenly
        indices = np.linspace(0, total_frames - 1, target_frames).astype(int)

    frames_data = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frames_data.append(np.zeros(17 * 2))  # Padding if frame read fails
            continue

        results = model(frame, verbose=False)

        # Get the keypoints of the first (most prominent) person
        if results and len(results[0].keypoints) > 0:
            # xyn = normalized [0-1] coordinates (17, 2)
            kpts = results[0].keypoints.xyn[0].cpu().numpy().flatten()
            frames_data.append(kpts)
        else:
            frames_data.append(np.zeros(17 * 2))  # Zero-fill if no person detected

    cap.release()
    return np.array(frames_data)


# --- Main Processing Loop ---

# Configuration

CLASSES = ["Walking", "Running"]

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def resolve_class_dir(data_dir, class_name):
    candidates = [
        class_name,
        class_name.lower(),
        class_name.upper(),
        class_name.capitalize(),
    ]
    for cand in candidates:
        cand_path = os.path.join(data_dir, cand)
        if os.path.isdir(cand_path):
            return cand
    for entry in os.listdir(data_dir):
        entry_path = os.path.join(data_dir, entry)
        if os.path.isdir(entry_path) and entry.lower() == class_name.lower():
            return entry
    raise FileNotFoundError(
        f"Class folder not found for '{class_name}' under {data_dir}"
    )


data_list = []

os.makedirs(OUTPUT_DIR, exist_ok=True)

for label_idx, label_name in enumerate(CLASSES):
    class_dir = resolve_class_dir(DATA_DIR, label_name)
    folder_path = os.path.join(DATA_DIR, class_dir)
    npy_files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]

    if npy_files:
        source_files = npy_files
        source_is_npy = True
    else:
        video_files = [
            f for f in os.listdir(folder_path) if f.lower().endswith(VIDEO_EXTS)
        ]
        source_files = video_files
        source_is_npy = False

    if not source_files:
        print(f"No .npy or video files found in {folder_path}, skipping.")
        continue

    for file_name in source_files:
        if source_is_npy:
            # Load (30, 34) array
            sequence = np.load(os.path.join(folder_path, file_name))
        else:
            video_path = os.path.join(folder_path, file_name)
            print(f"Extraing landmarks from {video_path}")
            sequence = extract_landmarks(video_path, target_frames=SEQUENCE_LENGTH)
        # Flatten each frame into rows for the CSV
        for frame_idx, landmarks in enumerate(sequence):
            row = {
                "sequence_id": f"{label_name}_{file_name}",
                "frame": frame_idx,
                "label": label_idx,
            }
            # Add each of the 34 coordinates as its own column (x0, y0, x1, y1...)
            for i, val in enumerate(landmarks):
                row[f"kpt_{i}"] = val
            data_list.append(row)


df = pd.DataFrame(data_list)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved {len(df)} rows to {OUTPUT_CSV}")
