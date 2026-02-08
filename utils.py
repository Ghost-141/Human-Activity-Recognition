# Add this import at the top
import cv2
import mediapipe as mp

from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import face_mesh_connections
from mediapipe.python.solutions import hands_connections


def draw_face_landmarks(frame, face_landmarks):
    # Iterate through each face detected
    for landmarks in face_landmarks:
        # Convert landmarks to a format drawing_utils understands
        # or simply draw them manually for more control:
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


def draw_hand_landmarks(frame, hand_landmarks):
    for landmarks in hand_landmarks:
        # Draw connections (lines)
        for connection in mp_hands.HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]

            cv2.line(
                frame,
                (int(start_lm.x * w), int(start_lm.y * h)),
                (int(end_lm.x * w), int(end_lm.y * h)),
                (255, 0, 0),
                2,
            )

        # Draw joints (dots)
        for lm in landmarks:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 0, 255), -1)


# ... inside the while loop after detection ...

# Draw Face
if face_result.face_landmarks:
    for face_lms in face_result.face_landmarks:
        # Drawing the points
        for lm in face_lms:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)

# Draw Hands
if hand_result.hand_landmarks:
    for hand_lms in hand_result.hand_landmarks:
        # Use MediaPipe's built-in utility for a cleaner look
        # This requires importing mp.solutions.drawing_utils as mp_drawing
        proto_lms = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
        proto_lms.landmark.extend(
            [
                mp.framework.formats.landmark_pb2.NormalizedLandmark(
                    x=lm.x, y=lm.y, z=lm.z
                )
                for lm in hand_lms
            ]
        )

        mp_drawing.draw_landmarks(
            frame,
            proto_lms,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp_drawing.get_default_landmark_style(),
            mp_drawing.get_default_connection_style(),
        )
