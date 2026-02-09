from ultralytics import YOLO
import cv2


model = YOLO("yolo11n-pose.pt")

cap = cv2.VideoCapture("dataset/PullUps/v_Pullup_g01_c01.avi")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    for result in results:
        print(result.keypoints.xyn.cpu().numpy()[0])
        print("-------------n----------------")
        print(len(result.keypoints.xyn.cpu().numpy()[0]))
    cv2.imshow("Inference Window", results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()
