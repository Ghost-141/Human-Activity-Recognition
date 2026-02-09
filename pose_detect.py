from collections import defaultdict, deque
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import supervision as sv

# ---------------------------
# Models
# ---------------------------
yolo_model = YOLO("yolo11m-pose.pt")
lstm_model = torch.jit.load("action_model_jit.pt")
lstm_model.eval()

# target_fps = 30, source_fps = 50
skip_rate = 50 / 30  # approx 1.66
frame_count = 0
last_processed_frame = 0

# ---------------------------
# Supervision tools
# ---------------------------
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Track-wise pose buffers (each person has their own deque)
pose_history = defaultdict(lambda: deque(maxlen=30))

# Optional: cache last action per track so label doesn't flicker
last_action = {}

cap = cv2.VideoCapture("videos/video_3.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count >= last_processed_frame + skip_rate:

        results = yolo_model(frame, verbose=False)
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
        kpts_xyn = r.keypoints.xyn.cpu().numpy()  # N x K x 2

        # ---------------------------
        # Per-person LSTM prediction
        # ---------------------------
        labels = []

        for i in range(len(detections)):
            tid = detections.tracker_id[i]
            if tid is None:
                labels.append("id: ?")
                continue

            # Flatten (K,2) -> (K*2,) for this person
            kpts_flat = kpts_xyn[i].reshape(-1).astype(np.float32)
            pose_history[tid].append(kpts_flat)

            action = last_action.get(tid, "…")

            if len(pose_history[tid]) == 30:
                seq = np.array(pose_history[tid], dtype=np.float32)  # (30, K*2)
                input_seq = torch.from_numpy(seq).unsqueeze(0)  # (1, 30, K*2)

                with torch.no_grad():
                    pred = lstm_model(input_seq)
                    class_idx = int(torch.argmax(pred, dim=1).item())

                # Map your classes here
                action = "Walking" if class_idx == 0 else "Running"
                last_action[tid] = action

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
