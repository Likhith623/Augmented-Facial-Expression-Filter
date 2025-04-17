import cv2
import numpy as np


filter_img = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED)

filter_width = int(np.linalg.norm(np.array([lx, ly]) - np.array([rx, ry])))

resized_filter = cv2.resize(
    filter_img,(filter_width, int(filter_width * filter_img.shape[0] / filter_img.shape[1]))
)


h_filter, w_filter, _ = resized_filter.shape

# Position (top left cordinates)
x = nx - w_filter // 2
y = ny - h_filter // 2


alpha = resized_filter[:, :, 3] / 255.0
color = resized_filter[:, :, :3]

#Where on the frame to paste
x1, x2 = max(0, x), min(frame.shape[1], x + w_filter)
y1, y2 = max(0, y), min(frame.shape[0], y + h_filter)

#Which part of the filter to copy(inside filter)
filter_x1 = max(0, -x)
filter_y1 = max(0, -y)


#overlay it on the webcam frame(pixel by pixel)
for c in range(3):
    frame[y:y+h_filter, x:x+w_filter, c] = (
        alpha * color[:, :, c] +
        (1 - alpha) * frame[y:y+h_filter, x:x+w_filter, c]
    )

