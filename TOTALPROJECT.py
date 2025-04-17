import cv2
import numpy as np
import mediapipe as mp
from fer import FER
horns_img = cv2.imread("angry.png", cv2.IMREAD_UNCHANGED)
cap_filter = cv2.imread("cap.png", cv2.IMREAD_UNCHANGED)
surprised_eyes_img = cv2.imread("glass.png", cv2.IMREAD_UNCHANGED)
smile_img = cv2.imread("hmouth.png", cv2.IMREAD_UNCHANGED)
mustache_img = cv2.imread("moustache.png", cv2.IMREAD_UNCHANGED)
neutral_face = cv2.imread("neutral.png", cv2.IMREAD_UNCHANGED)
mouth_filter = cv2.imread("ssmouth.png", cv2.IMREAD_UNCHANGED)
sunglasses_img = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED)
teardrop_img = cv2.imread("teardrop.png", cv2.IMREAD_UNCHANGED)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
emotion_detector = FER()
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    emotion = None
    try:
        detected_faces = emotion_detector.detect_emotions(rgb)
        if detected_faces:
            emotions_dict = detected_faces[0]['emotions']
            emotion_name, score = max(emotions_dict.items(), key=lambda x: x[1])
            if score > 0.4:
                emotion = emotion_name
                cv2.putText(frame, f"Emotion: {emotion} ({score:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    except Exception as e:
        print(f"Emotion detection error: {str(e)}")
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        def get_point(index):
            landmark = face_landmarks.landmark[index]
            return int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
        try:
            if emotion == 'happy':
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
                fx1, fy1 = max(0, -x), max(0, -y)
                fx2, fy2 = fx1 + (x2 - x1), fy1 + (y2 - y1)

                if fy2 <= resized_sunglasses.shape[0] and fx2 <= resized_sunglasses.shape[1]:
                    alpha = resized_sunglasses[fy1:fy2, fx1:fx2, 3] / 255.0 * 0.7
                    color = resized_sunglasses[fy1:fy2, fx1:fx2, :3]
                    for c in range(3):
                        frame[y1:y2, x1:x2, c] = alpha * color[:, :, c] + (1 - alpha) * frame[y1:y2, x1:x2, c]

                # Smile
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
                fx1_s, fy1_s = max(0, -x_smile), max(0, -y_smile)
                fx2_s, fy2_s = fx1_s + (x2_s - x1_s), fy1_s + (y2_s - y1_s)

                if fy2_s <= resized_smile.shape[0] and fx2_s <= resized_smile.shape[1]:
                    alpha_smile = resized_smile[fy1_s:fy2_s, fx1_s:fx2_s, 3] / 255.0
                    color_smile = resized_smile[fy1_s:fy2_s, fx1_s:fx2_s, :3]
                    for c in range(3):
                        frame[y1_s:y2_s, x1_s:x2_s, c] = alpha_smile * color_smile[:, :, c] + (1 - alpha_smile) * frame[y1_s:y2_s, x1_s:x2_s, c]

            elif emotion == 'angry':
                left_temple, right_temple, top_forehead = get_point(71), get_point(301), get_point(10)
                horns_width = int(np.linalg.norm(np.array(left_temple) - np.array(right_temple)) * 1.8)
                horns_height = int(horns_width * horns_img.shape[0] / horns_img.shape[1])
                resized_horns = cv2.resize(horns_img, (horns_width, horns_height))

                horn_x = top_forehead[0] - horns_width // 2
                horn_y = top_forehead[1] - horns_height
                x1_h, y1_h = max(0, horn_x), max(0, horn_y)
                x2_h, y2_h = min(frame.shape[1], horn_x + horns_width), min(frame.shape[0], horn_y + horns_height)
                fx1_h, fy1_h = max(0, -horn_x), max(0, -horn_y)
                fx2_h, fy2_h = fx1_h + (x2_h - x1_h), fy1_h + (y2_h - y1_h)

                if fy2_h <= resized_horns.shape[0] and fx2_h <= resized_horns.shape[1]:
                    alpha_horns = resized_horns[fy1_h:fy2_h, fx1_h:fx2_h, 3] / 255.0
                    color_horns = resized_horns[fy1_h:fy2_h, fx1_h:fx2_h, :3]
                    for c in range(3):
                        frame[y1_h:y2_h, x1_h:x2_h, c] = alpha_horns * color_horns[:, :, c] + (1 - alpha_horns) * frame[y1_h:y2_h, x1_h:x2_h, c]

                # Mustache
                nose_tip = get_point(4)
                upper_lip = get_point(13)
                mustache_center_x = nose_tip[0]
                mustache_center_y = (nose_tip[1] + upper_lip[1]) // 2
                face_width = int(np.linalg.norm(np.array(left_temple) - np.array(right_temple)))
                mustache_width = int(face_width * 0.9)
                mustache_height = int(mustache_width * mustache_img.shape[0] / mustache_img.shape[1])
                resized_mustache = cv2.resize(mustache_img, (mustache_width, mustache_height))
                x_mustache = mustache_center_x - mustache_width // 2
                y_mustache = mustache_center_y - mustache_height // 2
                x1_m, y1_m = max(0, x_mustache), max(0, y_mustache)
                x2_m, y2_m = min(frame.shape[1], x_mustache + mustache_width), min(frame.shape[0], y_mustache + mustache_height)
                fx1_m, fy1_m = max(0, -x_mustache), max(0, -y_mustache)
                fx2_m, fy2_m = fx1_m + (x2_m - x1_m), fy1_m + (y2_m - y1_m)

                if fy2_m <= resized_mustache.shape[0] and fx2_m <= resized_mustache.shape[1]:
                    alpha_m = resized_mustache[fy1_m:fy2_m, fx1_m:fx2_m, 3] / 255.0
                    color_m = resized_mustache[fy1_m:fy2_m, fx1_m:fx2_m, :3]
                    for c in range(3):
                        frame[y1_m:y2_m, x1_m:x2_m, c] = alpha_m * color_m[:, :, c] + (1 - alpha_m) * frame[y1_m:y2_m, x1_m:x2_m, c]

            elif emotion == "surprise":
                # --- Eyes (Surprised Sunglasses) ---
                left_eye_outer = get_point(33)
                right_eye_outer = get_point(263)
                left_eye_bottom = get_point(145)
                right_eye_bottom = get_point(374)

                eye_center_x = (left_eye_outer[0] + right_eye_outer[0]) // 2
                eye_center_y = (left_eye_bottom[1] + right_eye_bottom[1]) // 2
                eye_width = int(np.linalg.norm(np.array(left_eye_outer) - np.array(right_eye_outer)))
                filter_width = int(eye_width * 2.0)
                filter_height = int(filter_width * surprised_eyes_img.shape[0] / surprised_eyes_img.shape[1])
                resized_eyes = cv2.resize(surprised_eyes_img, (filter_width, filter_height))

                x = eye_center_x - filter_width // 2
                y = eye_center_y - filter_height // 2
                x1, x2 = max(0, x), min(frame.shape[1], x + filter_width)
                y1, y2 = max(0, y), min(frame.shape[0], y + filter_height)
                fx1, fy1 = max(0, -x), max(0, -y)
                fx2, fy2 = fx1 + (x2 - x1), fy1 + (y2 - y1)

                if fy2 <= resized_eyes.shape[0] and fx2 <= resized_eyes.shape[1]:
                    alpha = resized_eyes[fy1:fy2, fx1:fx2, 3] / 255.0
                    color = resized_eyes[fy1:fy2, fx1:fx2, :3]
                    for c in range(3):
                        frame[y1:y2, x1:x2, c] = alpha * color[:, :, c] + (1 - alpha) * frame[y1:y2, x1:x2, c]

                # --- Mouth (Surprised Mouth) ---
                left_mouth = get_point(78)
                right_mouth = get_point(308)
                top_lip = get_point(13)
                bottom_lip = get_point(14)

                mouth_center_x = (left_mouth[0] + right_mouth[0]) // 2
                mouth_center_y = (top_lip[1] + bottom_lip[1]) // 2
                mouth_width = int(np.linalg.norm(np.array(left_mouth) - np.array(right_mouth)) * 1.7)
                mouth_height = int((bottom_lip[1] - top_lip[1]) * 3)
                resized_mouth = cv2.resize(mouth_filter, (mouth_width, mouth_height))

                x_mouth = mouth_center_x - mouth_width // 2
                y_mouth = mouth_center_y - mouth_height // 2
                x1_m, x2_m = max(0, x_mouth), min(frame.shape[1], x_mouth + mouth_width)
                y1_m, y2_m = max(0, y_mouth), min(frame.shape[0], y_mouth + mouth_height)
                fx1_m, fy1_m = max(0, -x_mouth), max(0, -y_mouth)
                fx2_m, fy2_m = fx1_m + (x2_m - x1_m), fy1_m + (y2_m - y1_m)

                if fy2_m <= resized_mouth.shape[0] and fx2_m <= resized_mouth.shape[1]:
                    alpha_mouth = resized_mouth[fy1_m:fy2_m, fx1_m:fx2_m, 3] / 255.0
                    color_mouth = resized_mouth[fy1_m:fy2_m, fx1_m:fx2_m, :3]
                    for c in range(3):
                        frame[y1_m:y2_m, x1_m:x2_m, c] = alpha_mouth * color_mouth[:, :, c] + (1 - alpha_mouth) * frame[
                                                                                                                  y1_m:y2_m,
                                                                                                                 x1_m:x2_m,
                                                                                                                c]
            elif emotion == 'sad':
                left_eye_bottom = get_point(145)
                right_eye_bottom = get_point(374)
                left_eye_outer = get_point(33)
                right_eye_outer = get_point(263)
                eye_width = int(np.linalg.norm(np.array(left_eye_outer) - np.array(right_eye_outer)))
                teardrop_width = int(eye_width * 0.4)
                teardrop_height = int(teardrop_width * teardrop_img.shape[0] / teardrop_img.shape[1])
                resized_teardrop = cv2.resize(teardrop_img, (teardrop_width, teardrop_height))

                for eye_bottom in [left_eye_bottom, right_eye_bottom]:
                    x_td = eye_bottom[0] - teardrop_width // 2
                    y_td = eye_bottom[1] + 10
                    x1_td, x2_td = max(0, x_td), min(frame.shape[1], x_td + teardrop_width)
                    y1_td, y2_td = max(0, y_td), min(frame.shape[0], y_td + teardrop_height)
                    fx1_td, fy1_td = max(0, -x_td), max(0, -y_td)
                    fx2_td, fy2_td = fx1_td + (x2_td - x1_td), fy1_td + (y2_td - y1_td)

                    if fy2_td <= resized_teardrop.shape[0] and fx2_td <= resized_teardrop.shape[1]:
                        alpha_td = resized_teardrop[fy1_td:fy2_td, fx1_td:fx2_td, 3] / 255.0
                        color_td = resized_teardrop[fy1_td:fy2_td, fx1_td:fx2_td, :3]
                        for c in range(3):
                            frame[y1_td:y2_td, x1_td:x2_td, c] = alpha_td * color_td[:, :, c] + \
                                                                 (1 - alpha_td) * frame[y1_td:y2_td, x1_td:x2_td, c]

            elif emotion == 'neutral':
                # === Apply Cap ===
                left_temple = get_point(71)
                right_temple = get_point(301)
                top_forehead = get_point(10)
                cap_width = int(np.linalg.norm(np.array(left_temple) - np.array(right_temple)) * 1.8)
                cap_height = int(cap_width * cap_filter.shape[0] / cap_filter.shape[1])
                resized_cap = cv2.resize(cap_filter, (cap_width, cap_height))

                cap_x = top_forehead[0] - cap_width // 2
                cap_y = top_forehead[1] - int(cap_height * 0.7)  # was 1.0 → now 0.7 to move it down
                x1_c, x2_c = max(0, cap_x), min(frame.shape[1], cap_x + cap_width)
                y1_c, y2_c = max(0, cap_y), min(frame.shape[0], cap_y + cap_height)
                fx1_c, fy1_c = max(0, -cap_x), max(0, -cap_y)
                fx2_c, fy2_c = fx1_c + (x2_c - x1_c), fy1_c + (y2_c - y1_c)

                if fy2_c <= resized_cap.shape[0] and fx2_c <= resized_cap.shape[1]:
                    alpha_c = resized_cap[fy1_c:fy2_c, fx1_c:fx2_c, 3] / 255.0
                    color_c = resized_cap[fy1_c:fy2_c, fx1_c:fx2_c, :3]
                    for c in range(3):
                        frame[y1_c:y2_c, x1_c:x2_c, c] = alpha_c * color_c[:, :, c] + (1 - alpha_c) * frame[y1_c:y2_c, x1_c:x2_c, c]

                # === Apply Neutral Face Filter ===
                left_eye_outer = get_point(33)
                right_eye_outer = get_point(263)
                left_mouth = get_point(78)
                right_mouth = get_point(308)

                face_center_x = (left_eye_outer[0] + right_eye_outer[0]) // 2
                face_center_y = (left_mouth[1] + right_eye_outer[1]) // 2
                face_width = int(np.linalg.norm(np.array(left_eye_outer) - np.array(right_eye_outer)) * 1.8)
                face_height = int(face_width * neutral_face.shape[0] / neutral_face.shape[1])
                resized_neutral = cv2.resize(neutral_face, (face_width, face_height))

                face_x = face_center_x - face_width // 2
                face_y = face_center_y - int(face_height * 0.7)  # was 0.5 → now 0.6 to nudge it up
                x1_n, x2_n = max(0, face_x), min(frame.shape[1], face_x + face_width)
                y1_n, y2_n = max(0, face_y), min(frame.shape[0], face_y + face_height)
                fx1_n, fy1_n = max(0, -face_x), max(0, -face_y)
                fx2_n, fy2_n = fx1_n + (x2_n - x1_n), fy1_n + (y2_n - y1_n)

                if fy2_n <= resized_neutral.shape[0] and fx2_n <= resized_neutral.shape[1]:
                    alpha_n = resized_neutral[fy1_n:fy2_n, fx1_n:fx2_n, 3] / 255.0
                    color_n = resized_neutral[fy1_n:fy2_n, fx1_n:fx2_n, :3]
                    for c in range(3):
                        frame[y1_n:y2_n, x1_n:x2_n, c] = alpha_n * color_n[:, :, c] + (1 - alpha_n) * frame[y1_n:y2_n, x1_n:x2_n, c]

        except Exception as e:
               print(f"Filter application error: {str(e)}")
    # Show frame
    cv2.imshow("Snapchat-Style Filters", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
