import os
import cv2
import time
import numpy as np
from collections import deque
import mediapipe as mp

# Environment Fixes for Linux
os.environ["QT_QPA_PLATFORM"] = "xcb"
from mediapipe.framework.formats import landmark_pb2

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# -----------------------------
# Tunables & State
# -----------------------------
HAND_FACE_DIST_THRESH = 0.5
YAW_THRESH = 0.015  # Sensitivity for head turning

# Timers State
cover_start_time = None
cover_duration = 0.0

yaw_start_time = None
yaw_duration = 0.0


def distance(a, b):
    return float(np.linalg.norm(a - b))


cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    model_complexity=2,
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.4,
) as holistic:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        t_now = time.time()
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        is_currently_covering = False
        is_currently_turning = False

        # 1. Logic for Hand Covering Face
        if results.face_landmarks and (
            results.left_hand_landmarks or results.right_hand_landmarks
        ):
            f = results.face_landmarks.landmark
            nose = np.array([f[1].x, f[1].y])
            # Eye distance for scaling
            eye_dist = distance(
                np.array([f[33].x, f[33].y]), np.array([f[263].x, f[263].y])
            )

            # Yaw Proxy (Nose horizontal offset from eye midpoint)
            eye_mid_x = (f[33].x + f[263].x) / 2.0
            yaw_val = abs(f[1].x - eye_mid_x) / eye_dist
            if yaw_val > YAW_THRESH:
                is_currently_turning = True

            for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
                if hand:
                    index_tip = np.array([hand.landmark[8].x, hand.landmark[8].y])
                    if distance(index_tip, nose) / eye_dist < HAND_FACE_DIST_THRESH:
                        is_currently_covering = True
                        break

        # 2. Timer Updates
        # Face Covering Timer
        if is_currently_covering:
            if cover_start_time is None:
                cover_start_time = t_now
            cover_duration = t_now - cover_start_time
        else:
            cover_start_time = None
            cover_duration = 0.0

        # Head Turning Timer
        if is_currently_turning:
            if yaw_start_time is None:
                yaw_start_time = t_now
            yaw_duration = t_now - yaw_start_time
        else:
            yaw_start_time = None
            yaw_duration = 0.0

        # 3. Drawing (Dots and Connections, NO MASK)
        # Draw Face Dots & Fine Lines
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )

        # Draw Pose Connections
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )

        # Draw Hand Connections
        for hand_lms in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand_lms:
                mp_drawing.draw_landmarks(frame, hand_lms, mp_holistic.HAND_CONNECTIONS)

        # 4. Display Timers
        # Background box for readability
        cv2.rectangle(frame, (10, 10), (350, 120), (0, 0, 0), -1)

        cv2.putText(
            frame,
            f"Covering Time: {cover_duration:.1f}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"Head Turn Time: {yaw_duration:.1f}s",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            1,
        )

        # Suspicion Logic based on duration
        if cover_duration > 3.0 or yaw_duration > 5.0:
            cv2.putText(
                frame,
                "SUSPICIOUS DURATION",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Behavior Analysis with Timers", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
