import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
IMAGE = os.path.join(ROOT, 'asset', 'test.png')

img = cv2.imread(IMAGE)

height, width = img.shape[:2]

dummy_img = np.zeros(img.shape, np.uint8)

img = cv2.resize(img, (0, 0), fx=1, fy=0.5)

blur = cv2.GaussianBlur(img, (15, 15), 0)

edges = cv2.Canny(img, 100, 200, 5)
edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

dummy_img[:height // 2, :] = blur
dummy_img[height // 2:, :] = edges

cv2.imshow('Frame', dummy_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
