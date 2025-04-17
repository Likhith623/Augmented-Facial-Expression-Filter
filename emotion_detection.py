#everything runs on CPU , not on gpu
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
from fer import FER
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
detector = FER()
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break
        results = detector.detect_emotions(frame)
        for result in results:
            (x, y, w, h) = result["box"]
            emotions = result["emotions"]
            top_emotion = max(emotions, key=emotions.get)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, top_emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed.")