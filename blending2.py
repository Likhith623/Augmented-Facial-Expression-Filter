import cv2
import numpy as np
import mediapipe as mp

filter_img = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED)


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            left_eye_outer = face_landmarks.landmark[33]
            right_eye_outer = face_landmarks.landmark[263]
            left_eye_bottom = face_landmarks.landmark[145]
            right_eye_bottom = face_landmarks.landmark[374]

            # Calculate eye center
            eye_center_x = int((left_eye_outer.x + right_eye_outer.x) / 2 * w)
            eye_center_y = int((left_eye_bottom.y + right_eye_bottom.y) / 2 * h)


            eye_width = int(np.linalg.norm(np.array([left_eye_outer.x * w, left_eye_outer.y * h]) - np.array([right_eye_outer.x * w, right_eye_outer.y * h])))
            filter_width = int(eye_width * 2.0)  # Scale to make it bigger
            filter_height = int(filter_width * filter_img.shape[0] / filter_img.shape[1])

            # Resize filter
            resized_filter = cv2.resize(filter_img, (filter_width, filter_height))
            h_filter, w_filter, _ = resized_filter.shape

            # Top-left corner for placing
            x1 = eye_center_x - w_filter // 2
            y1 = eye_center_y - h_filter // 2 - 10
            x2 = x1 + w_filter
            y2 = y1 + h_filter

            # Boundary checks
            trans = 0.7
            if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
                alpha = resized_filter[:, :, 3] / 255.0 * trans
                color = resized_filter[:, :, :3]

                for c in range(3):
                    frame[y1:y2, x1:x2, c] = (
                        alpha * color[:, :, c] +
                        (1 - alpha) * frame[y1:y2, x1:x2, c]
                    )

    cv2.imshow("Snapchat Filter", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
