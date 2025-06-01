import os

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    h, w = frame.shape[:2]

    dummy_frame = np.zeros(frame.shape, np.uint8)

    frame = cv2.resize(frame, (0, 0), fx=1, fy=0.5)

    blur = cv2.GaussianBlur(frame, (15, 15), 1)

    edges = cv2.Canny(frame, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    dummy_frame[:h // 2, :] = blur
    dummy_frame[h // 2:, :] = edges

    cv2.imshow('Frame', dummy_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()