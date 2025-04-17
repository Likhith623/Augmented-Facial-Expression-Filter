import cv2
import numpy as np
import mediapipe as mp
from fer import FER

# Load filters with alpha channel
sunglasses_img = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED)
smile_img = cv2.imread("hmouth.png", cv2.IMREAD_UNCHANGED)


# Initialize MediaPipe and FER
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
emotion_detector = FER()

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect emotion
    emotion_result = emotion_detector.top_emotion(rgb)

    # Detect facial landmarks
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks and emotion_result:
        face_landmarks = results.multi_face_landmarks[0]

        def get_point(index):
            landmark = face_landmarks.landmark[index]
            return int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])

        # === Happy Filter ===
        if emotion_result[0] == 'happy':
            # Sunglasses
            left_eye_outer = get_point(33)
            right_eye_outer = get_point(263)
            left_eye_bottom = get_point(145)
            right_eye_bottom = get_point(374)

            eye_center_x = (left_eye_outer[0] + right_eye_outer[0]) // 2
            eye_center_y = (left_eye_bottom[1] + right_eye_bottom[1]) // 2
            eye_width = int(np.linalg.norm(np.array(left_eye_outer) - np.array(right_eye_outer)))


            filter_width = int(eye_width * 2.0)
            filter_height = int(filter_width * sunglasses_img.shape[0] / sunglasses_img.shape[1])
            resized_sunglasses = cv2.resize(sunglasses_img, (filter_width, filter_height))

            x = eye_center_x - filter_width // 2
            y = eye_center_y - filter_height // 2 - 20

            x1, x2 = max(0, x), min(frame.shape[1], x + filter_width)
            y1, y2 = max(0, y), min(frame.shape[0], y + filter_height)

            filter_x1 = max(0, -x)
            filter_y1 = max(0, -y)
            filter_x2 = filter_x1 + (x2 - x1)
            filter_y2 = filter_y1 + (y2 - y1)

            transparency = 0.7
            alpha = (resized_sunglasses[filter_y1:filter_y2, filter_x1:filter_x2, 3] / 255.0) * transparency
            color = resized_sunglasses[filter_y1:filter_y2, filter_x1:filter_x2, :3]

            for c in range(3):
                frame[y1:y2, x1:x2, c] = (
                        alpha * color[:, :, c] + (1 - alpha) * frame[y1:y2, x1:x2, c]
                )

            # Smiling Mouth
            upper_lip = get_point(13)
            lower_lip = get_point(14)

            mouth_center_x = (upper_lip[0] + lower_lip[0]) // 2
            mouth_center_y = (upper_lip[1] + lower_lip[1]) // 2
            mouth_width = int(eye_width * 1.0)
            mouth_height = int(mouth_width * smile_img.shape[0] / smile_img.shape[1])
            resized_smile = cv2.resize(smile_img, (mouth_width, mouth_height))

            x_smile = mouth_center_x - mouth_width // 2
            y_smile = mouth_center_y - mouth_height // 2 + 10

            x1_s, x2_s = max(0, x_smile), min(frame.shape[1], x_smile + mouth_width)
            y1_s, y2_s = max(0, y_smile), min(frame.shape[0], y_smile + mouth_height)

            fx1_s = max(0, -x_smile)
            fy1_s = max(0, -y_smile)
            fx2_s = fx1_s + (x2_s - x1_s)
            fy2_s = fy1_s + (y2_s - y1_s)

            alpha_smile = resized_smile[fy1_s:fy2_s, fx1_s:fx2_s, 3] / 255.0
            color_smile = resized_smile[fy1_s:fy2_s, fx1_s:fx2_s, :3]

            for c in range(3):
                frame[y1_s:y2_s, x1_s:x2_s, c] = (
                    alpha_smile * color_smile[:, :, c] + (1 - alpha_smile) * frame[y1_s:y2_s, x1_s:x2_s, c]
                )


    # Show result
    cv2.imshow("Emotion Filter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
