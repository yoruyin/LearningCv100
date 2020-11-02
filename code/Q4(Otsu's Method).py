import cv2
import numpy as np


def BGR2Grayscale(img):
    r = img[:, :, 2].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 0].copy()

    img2 = 0.2126*r + 0.7152*g + 0.0722*b

    return img2


def otsu_binarization(img,th=128):
    max_sigma = 0
    max_t = 0

    for _t in range(1,255):
        v0 = img[np.where(img<_t)]
        m0 = np.mean(v0) if len(v0)>0 else 0.
        w0 = len(v0)/(H*W)
        v1 = img[np.where(img>=_t)]
        m1 = np.mean(v1) if len(v1)>0 else 0.
        w1 = len(v1)/(H*W)
        sigma = w0*w1*((m0-m1)**2)
        if sigma > max_sigma:
            max_sigma = sigma
            max_t = _t

    th = max_t
    img[img<th] =0
    img[img>=th] = 255
    return img

img = cv2.imread("assert/imori.jpg").astype(np.float)
H , W, C =img.shape
img = BGR2Grayscale(img)
img = otsu_binarization(img)
cv2.namedWindow("imori",cv2.WINDOW_AUTOSIZE)
cv2.imshow("imori",img)
cv2.waitKey(0)
