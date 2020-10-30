import cv2
import numpy as np


def coler_dicrease(img):
    img = img//64*64+32
    return img

img = cv2.imread("assert/imori.jpg").astype(np.float32)
img = coler_dicrease(img).astype(np.uint8)

cv2.imshow("imori",img)
cv2.waitKey(0)