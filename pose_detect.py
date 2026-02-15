from collections import defaultdict, deque
from ultralytics import YOLO
import torch
import json
import cv2
import numpy as np
import supervision as sv
import torch.nn.functional as F

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------
# Models
# ---------------------------
yolo_model = YOLO("yolo11m-pose.pt")
lstm_model = torch.jit.load("action_model_jit.pt").to(device)
lstm_model.eval()

# target_fps = 30, source_fps = 50
skip_rate = 50 / 30  # approx 1.66
frame_count = 0
last_processed_frame = 0
UNKNOWN_CONF_THR = 0.40


# ---------------------------
# Helper with class
# ---------------------------
# COCO indices for YOLO pose
# Load label mapping from training

label2id = json.load(open("final_data/labels.json"))
id2label = {v: k for k, v in label2id.items()}

L_SH, R_SH = 5, 6
L_HIP, R_HIP = 11, 12


def normalize_xy_window(xy):  # xy: (T,17,2)
    root = (xy[:, L_HIP] + xy[:, R_HIP]) / 2.0
    xy = xy - root[:, None, :]

    mid_sh = (xy[:, L_SH] + xy[:, R_SH]) / 2.0
    mid_hip = (xy[:, L_HIP] + xy[:, R_HIP]) / 2.0
    scale = np.linalg.norm(mid_sh - mid_hip, axis=-1)
    scale = np.clip(scale, 1e-4, None)

    return xy / scale[:, None, None]


# ---------------------------
# Supervision tools
# ---------------------------
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Track-wise pose buffers (each person has their own deque)
pose_xy = defaultdict(lambda: deque(maxlen=30))  # stores (17,2)
pose_cf = defaultdict(lambda: deque(maxlen=30))  # stores (17,)
last_action = {}

DEBUG_TID = None

cap = cv2.VideoCapture("videos/video_3.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count - last_processed_frame < skip_rate:
        continue

    last_processed_frame = frame_count
    results = yolo_model(frame, conf=0.4, verbose=False)
    r = results[0]

    # If no detections, just show frame
    if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
        cv2.imshow("Action Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # ---------------------------
    # Build supervision detections (boxes/conf/class)
    # ---------------------------
    detections = sv.Detections.from_ultralytics(r)

    # IMPORTANT: ByteTrack expects float xyxy and confidence in detections
    # Update tracker to attach tracker_id to each detection
    detections = tracker.update_with_detections(detections)

    # Keypoints normalized xy for all persons: shape (N, K, 2)
    # (Ultralytics order aligns boxes <-> keypoints)
    kpts_xyn = r.keypoints.xyn.cpu().numpy()  # (N,17,2)
    kpts_conf = r.keypoints.conf.cpu().numpy()  # (N,17)

    # ---------------------------
    # Per-person LSTM prediction
    # ---------------------------
    labels = []

    for i in range(len(detections)):
        tid = detections.tracker_id[i]
        if DEBUG_TID is None and tid is not None:
            DEBUG_TID = tid
        if tid is None:
            labels.append("id: ?")
            continue

        xy = kpts_xyn[i].astype(np.float32)  # (17,2)
        cf = kpts_conf[i].astype(np.float32)  # (17,)

        pose_xy[tid].append(xy)
        pose_cf[tid].append(cf)

        action = last_action.get(tid, "…")

        if len(pose_xy[tid]) == 30:
            w_xy = np.stack(pose_xy[tid], axis=0)  # (30,17,2)
            w_cf = np.stack(pose_cf[tid], axis=0)  # (30,17)

            # SAME preprocessing as training
            w_xy = normalize_xy_window(w_xy)  # (30,17,2)
            xy_flat = w_xy.reshape(30, -1)  # (30,34)
            cf_flat = w_cf.reshape(30, -1)  # (30,17)

            vel = np.zeros_like(xy_flat)
            vel[1:] = xy_flat[1:] - xy_flat[:-1]  # (30,34)

            feat = np.concatenate([xy_flat, cf_flat, vel], axis=1)  # (30,85)

            input_seq = torch.from_numpy(feat).unsqueeze(0).to(device)  # (1,30,85)

            with torch.no_grad():
                pred = lstm_model(input_seq)
                probs = F.softmax(pred, dim=1).detach().cpu().numpy()[0]
                class_idx = int(torch.argmax(pred, dim=1).item())
                max_prob = float(np.max(probs))

            if max_prob < UNKNOWN_CONF_THR:
                action = "idle/unknown"
                print(f"tid={tid} probs={probs} max_prob={max_prob:.3f} => {action}")
            else:
                action = id2label[class_idx]
                last_action[tid] = action
                print(f"tid={tid} probs={probs} max_prob={max_prob:.3f} => {action}")

        labels.append(f"id:{tid} {action}")

    # ---------------------------
    # Draw boxes + labels
    # ---------------------------
    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
    annotated = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels
    )

    cv2.imshow("Action Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
