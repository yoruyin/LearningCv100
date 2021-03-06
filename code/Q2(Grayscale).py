import cv2
import numpy as np


def BGR2Grayscale(img):
    r = img[:, :, 2].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 0].copy()

    img2 = 0.2126*r + 0.7152*g + 0.0722*b

    return img2


img = cv2.imread("assert/imori.jpg").astype(np.float)
img = BGR2Grayscale(img).astype(np.uint8)
cv2.namedWindow("imori",cv2.WINDOW_AUTOSIZE)
cv2.imshow("imori",img)
cv2.waitKey(0)
cv2.destroyWindow()

