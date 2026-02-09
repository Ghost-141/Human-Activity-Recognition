import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ---------------- Config ----------------
DATA_DIR = "/home/imtiaz/Documents/GitHub/activity_det/dataset"
OUTPUT_CSV = "/home/imtiaz/Documents/GitHub/activity_det/all_pose_data.csv"
MODEL_PATH = "yolo11m-pose.pt"

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
K = 17
CONF_THR = 0.20
MIN_GOOD_JOINTS = 6
# ---------------------------------------

model = YOLO(MODEL_PATH)


def extract_people(results):
    """Extract all people keypoints + track IDs for a frame."""
    if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
        return [], [], None

    kxy = results[0].keypoints.xyn  # (P,17,2)
    kcf = results[0].keypoints.conf  # (P,17)
    ids = None
    if results[0].boxes is not None and results[0].boxes.id is not None:
        ids = results[0].boxes.id  # (P,)
    return kxy, kcf, ids


def forward_fill(seq, valid):
    """Forward + backward fill missing frames for inspectable CSV."""
    last = None
    for i in range(len(seq)):
        if valid[i] == 1:
            last = seq[i].copy()
        elif last is not None:
            seq[i] = last

    nxt = None
    for i in range(len(seq) - 1, -1, -1):
        if valid[i] == 1:
            nxt = seq[i].copy()
        elif nxt is not None:
            seq[i] = nxt
    return seq


def iter_class_folders(data_dir):
    for name in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, name)
        if os.path.isdir(p):
            yield name, p


def iter_videos(folder):
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(VIDEO_EXTS):
                yield os.path.join(root, f)


def process_video(video_path, class_name):
    cap = cv2.VideoCapture(video_path)
    rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, verbose=False)
        kxy, kcf, ids = extract_people(results)

        if len(kxy) == 0:
            frame_idx += 1
            continue

        for p in range(len(kxy)):
            frame_valid = 0
            pose = np.zeros((K, 3), dtype=np.float32)

            if int((kcf[p] > CONF_THR).sum()) >= MIN_GOOD_JOINTS:
                frame_valid = 1
                pose[:, 0:2] = kxy[p].cpu().numpy().astype(np.float32)
                pose[:, 2] = kcf[p].cpu().numpy().astype(np.float32)

            track_id = -1
            if ids is not None:
                track_id = int(ids[p].item())

            row = {
                "class": class_name,
                "video": os.path.basename(video_path),
                "frame": frame_idx,
                "track_id": track_id,
                "frame_valid": frame_valid,
            }

            for j in range(K):
                row[f"kpt{j}_x"] = float(pose[j, 0])
                row[f"kpt{j}_y"] = float(pose[j, 1])
                row[f"kpt{j}_conf"] = float(pose[j, 2])

            rows.append(row)
        frame_idx += 1

    cap.release()
    if hasattr(model, "reset"):
        model.reset()

    if not rows:
        return []

    df = pd.DataFrame(rows)

    # Fill missing frames per track_id (keep frame_valid column unchanged)
    cols = [f"kpt{j}_{c}" for j in range(K) for c in ("x", "y", "conf")]
    filled_groups = []
    for _, g in df.sort_values("frame").groupby("track_id", sort=False):
        seq = g[cols].to_numpy(dtype=np.float32, copy=True).reshape(-1, K, 3)
        valid = g["frame_valid"].to_numpy(dtype=np.int32)
        seq = forward_fill(seq, valid)
        g.loc[:, cols] = seq.reshape(len(g), K * 3)
        filled_groups.append(g)
    df = pd.concat(filled_groups, ignore_index=True)

    return df.to_dict("records")


def main():
    all_rows = []
    video_count = 0

    for class_name, class_path in iter_class_folders(DATA_DIR):
        videos = list(iter_videos(class_path))
        if not videos:
            print(f"[SKIP] No videos in {class_path}")
            continue

        for vp in videos:
            print(f"[INFO] Processing {vp}")
            rows = process_video(vp, class_name)
            all_rows.extend(rows)
            video_count += 1

    if not all_rows:
        print("No data collected.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n==============================")
    print(f"Processed videos : {video_count}")
    print(f"Total frames     : {len(df)}")
    print(f"Saved CSV        : {OUTPUT_CSV}")
    print("==============================\n")


if __name__ == "__main__":
    main()
