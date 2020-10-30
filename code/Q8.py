import cv2
import numpy as np


def average_pooling(img, G=8):
    img2 = img.copy()

    Nh = int(H / G)
    Nw = int(W / G)

    for y in range(Nh):
        for x in range(Nw):
            for c in range(C):
                img2[G * y:G * (y + 1), G * x:G * (x + 1), c] = np.max(img2[G * y:G * (y + 1), G * x:G * (x + 1), c])

    return img2


img = cv2.imread("assert/imori.jpg").astype(np.float32)
H, W, C = img.shape
print(H, W, C)
img2 = average_pooling(img, 8).astype(np.uint8)

cv2.imshow("imori", img2)
cv2.waitKey(0)
