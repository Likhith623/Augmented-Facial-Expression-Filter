import cv2
import numpy as np
import mediapipe as mp
from fer import FER


# Load filter images with alpha channel
horns_img = cv2.imread("angry.png", cv2.IMREAD_UNCHANGED)
mustache_img = cv2.imread("moustache.png", cv2.IMREAD_UNCHANGED)
smile_img = cv2.imread("ssmouth.png", cv2.IMREAD_UNCHANGED)
sunglasses_img = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED)


# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize FER emotion detector
emotion_detector = FER()

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect emotion with robust error handling
    emotion = None
    try:
        # Get all detected faces and emotions
        detected_faces = emotion_detector.detect_emotions(rgb)

        # Check if any faces were detected
        if detected_faces and len(detected_faces) > 0:
            # Get the emotions for the first face
            emotions_dict = detected_faces[0]['emotions']

            # Find the emotion with the highest score
            max_emotion = max(emotions_dict.items(), key=lambda x: x[1])
            emotion_name, score = max_emotion

            if score > 0.2:  # Lower threshold for better detection
                emotion = emotion_name
                # Display detected emotion
                cv2.putText(frame, f"Emotion: {emotion} ({score:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    except Exception as e:
        print(f"Emotion detection error: {str(e)}")

    # Detect facial landmarks
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]


        def get_point(index):
            landmark = face_landmarks.landmark[index]
            return int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])


        # Apply filters based on emotion
        if emotion == 'happy':
            try:
                # === Sunglasses ===
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

                fx1 = max(0, -x)
                fy1 = max(0, -y)
                fx2 = fx1 + (x2 - x1)
                fy2 = fy1 + (y2 - y1)

                if x1 < x2 and y1 < y2 and fx1 < fx2 and fy1 < fy2 and fy2 <= resized_sunglasses.shape[0] and fx2 <= \
                        resized_sunglasses.shape[1]:
                    tran = 0.7
                    alpha = resized_sunglasses[fy1:fy2, fx1:fx2, 3] / 255.0 * tran
                    color = resized_sunglasses[fy1:fy2, fx1:fx2, :3]

                    for c in range(3):
                        frame[y1:y2, x1:x2, c] = alpha * color[:, :, c] + (1 - alpha) * frame[y1:y2, x1:x2, c]

                # === Smiling Mouth ===
                upper_lip = get_point(13)
                lower_lip = get_point(14)

                mouth_center_x = (upper_lip[0] + lower_lip[0]) // 2
                mouth_center_y = (upper_lip[1] + lower_lip[1]) // 2
                mouth_width = int(eye_width * 0.9)
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

                if x1_s < x2_s and y1_s < y2_s and fx1_s < fx2_s and fy1_s < fy2_s and fy2_s <= resized_smile.shape[
                    0] and fx2_s <= resized_smile.shape[1]:
                    alpha_smile = resized_smile[fy1_s:fy2_s, fx1_s:fx2_s, 3] / 255.0
                    color_smile = resized_smile[fy1_s:fy2_s, fx1_s:fx2_s, :3]

                    for c in range(3):
                        frame[y1_s:y2_s, x1_s:x2_s, c] = alpha_smile * color_smile[:, :, c] + (1 - alpha_smile) * frame[
                                                                                                                  y1_s:y2_s,
                                                                                                                  x1_s:x2_s,
                                                                                                                  c]
            except Exception as e:
                print(f"Error applying happy filter: {str(e)}")

        elif emotion == 'angry':
            try:
                # === Horns ===
                left_temple = get_point(71)
                right_temple = get_point(301)
                top_forehead = get_point(10)

                horns_width = int(np.linalg.norm(np.array(left_temple) - np.array(right_temple)) * 1.8)
                horns_height = int(horns_width * horns_img.shape[0] / horns_img.shape[1])
                resized_horns = cv2.resize(horns_img, (horns_width, horns_height))

                horn_x = top_forehead[0] - horns_width // 2
                horn_y = top_forehead[1] - horns_height

                x1_h = max(0, horn_x)
                y1_h = max(0, horn_y)
                x2_h = min(frame.shape[1], horn_x + horns_width)
                y2_h = min(frame.shape[0], horn_y + horns_height)

                fx1_h = max(0, -horn_x)
                fy1_h = max(0, -horn_y)
                fx2_h = fx1_h + (x2_h - x1_h)
                fy2_h = fy1_h + (y2_h - y1_h)

                if x1_h < x2_h and y1_h < y2_h and fx1_h < fx2_h and fy1_h < fy2_h and fy2_h <= resized_horns.shape[
                    0] and fx2_h <= resized_horns.shape[1]:
                    alpha_horns = resized_horns[fy1_h:fy2_h, fx1_h:fx2_h, 3] / 255.0
                    color_horns = resized_horns[fy1_h:fy2_h, fx1_h:fx2_h, :3]

                    for c in range(3):
                        frame[y1_h:y2_h, x1_h:x2_h, c] = alpha_horns * color_horns[:, :, c] + (1 - alpha_horns) * frame[
                                                                                                                  y1_h:y2_h,
                                                                                                                  x1_h:x2_h,
                                                                                                                  c]

                # === Mustache ===
                # Get nose and mouth landmarks
                nose_tip = get_point(4)
                upper_lip = get_point(13)

                # Determine mustache position
                mustache_center_x = nose_tip[0]
                mustache_center_y = (nose_tip[1] + upper_lip[1]) // 2

                # Calculate mustache dimensions based on face
                face_width = int(np.linalg.norm(np.array(left_temple) - np.array(right_temple)))
                # Calculate mustache dimensions based on face
                mustache_scale = 2  # Increase to make mustache larger
                mustache_width = int(face_width * 0.6 * mustache_scale)  # Scale width
                mustache_height = int(
                    mustache_width * mustache_img.shape[0] / mustache_img.shape[1])  # Maintain aspect ratio

                # Resize mustache image
                resized_mustache = cv2.resize(mustache_img, (mustache_width, mustache_height))

                # Calculate placement coordinates
                x_mustache = mustache_center_x - mustache_width // 2
                y_mustache = mustache_center_y - mustache_height // 2

                # Define placement regions
                x1_m = max(0, x_mustache)
                y1_m = max(0, y_mustache)
                x2_m = min(frame.shape[1], x_mustache + mustache_width)
                y2_m = min(frame.shape[0], y_mustache + mustache_height)

                # Define corresponding regions in the filter image
                fx1_m = max(0, -x_mustache)
                fy1_m = max(0, -y_mustache)
                fx2_m = fx1_m + (x2_m - x1_m)
                fy2_m = fy1_m + (y2_m - y1_m)

                # Apply mustache overlay
                if (x1_m < x2_m and y1_m < y2_m and
                        fx1_m < fx2_m and fy1_m < fy2_m and
                        fy2_m <= resized_mustache.shape[0] and
                        fx2_m <= resized_mustache.shape[1]):

                    alpha_mustache = resized_mustache[fy1_m:fy2_m, fx1_m:fx2_m, 3] / 255.0
                    color_mustache = resized_mustache[fy1_m:fy2_m, fx1_m:fx2_m, :3]

                    for c in range(3):
                        frame[y1_m:y2_m, x1_m:x2_m, c] = (
                                alpha_mustache * color_mustache[:, :, c] +
                                (1 - alpha_mustache) * frame[y1_m:y2_m, x1_m:x2_m, c]
                        )
            except Exception as e:
                print(f"Error applying angry filter: {str(e)}")

    # Show output
    cv2.imshow("Emotion Filter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


