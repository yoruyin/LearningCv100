import cv2
import numpy as np


def BGR2HSV(_img):
    img = _img.copy() / 255.
    hsv = np.zeros_like(img, np.float32)

    max_v = np.max(img, axis=2).copy()
    min_v = np.min(img, axis=2).copy()
    min_arg = np.argmin(img, axis=2)

    # H
    ## Min == Max
    hsv[..., 0][np.where(max_v == min_v)] = 0
    ## Min == B
    ind = np.where(min_arg == 0)
    hsv[..., 0][ind] = 60 * (img[..., 1][ind] - img[..., 2][ind]) / (max_v[ind] - min_v[ind]) + 60
    ## Min == R
    ind = np.where(min_arg == 1)
    hsv[..., 0][ind] = 60 * (img[..., 0][ind] - img[..., 1][ind]) / (max_v[ind] - min_v[ind]) + 180
    ## Min == G
    ind = np.where(min_arg == 2)
    hsv[..., 0][ind] = 60 * (img[..., 2][ind] - img[..., 0][ind]) / (max_v[ind] - min_v[ind]) + 300

    # S
    hsv[..., 1] = max_v.copy() - min_v.copy()

    # V
    hsv[..., 2] = max_v.copy()

    return hsv


def HSV2BGR(_img, hsv):
    img = _img.copy() / 255.

    max_v = np.max(img, axis=2).copy()
    min_v = np.min(img, axis=2).copy()

    img2 = np.zeros_like(img, np.float32)

    H = hsv[..., 0]
    S = hsv[..., 1]
    V = hsv[..., 2]

    C = S
    HH = H / 60.
    X = C * (1 - np.abs(HH % 2 - 1))
    Z = np.zeros_like(H)

    vals = [[Z, X, C], [Z, C, X], [X, C, Z], [C, X, Z], [C, Z, X], [X, Z, C]]
    for i in range(6):
        ind = np.where((i <= HH) & (HH < (i + 1)))
        img2[..., 0][ind] = (V - C)[ind] + vals[i][0][ind]
        img2[..., 1][ind] = (V - C)[ind] + vals[i][1][ind]
        img2[..., 2][ind] = (V - C)[ind] + vals[i][2][ind]

    img2[np.where(max_v == min_v)] = 0
    img2 = np.clip(img2, 0, 1)
    img2 = (img2 * 255)

    return img2


img = cv2.imread("assert/imori.jpg").astype(np.float32)

hsv = BGR2HSV(img)
hsv[..., 0] = (hsv[..., 0] + 180) % 360

img2 = HSV2BGR(img, hsv)
img2 = img2.astype(np.uint8)

cv2.namedWindow("imori", cv2.WINDOW_AUTOSIZE)
cv2.imshow("imori", img2)
cv2.waitKey(0)