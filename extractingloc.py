h, w, _ = frame.shape  # Get frame size
# Get key facial points

# Left and right eye corners
#landmark.x = left to right across face
#landmark.y = top to bottom across face
#lanskmark.z = depth (how far away from the camera)
lx = int(face_landmarks.landmark[33].x * w)
ly = int(face_landmarks.landmark[33].y * h)
rx = int(face_landmarks.landmark[263].x * w)
ry = int(face_landmarks.landmark[263].y * h)
