import cv2
import numpy as np


def median_filter(img, K_size=3):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape

    ## Zero padding
    pad = K_size // 2
    img2 = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    img2[pad:pad + H, pad:pad + W] = img.copy().astype(np.float)

    tmp = img2.copy()

    ## filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                img2[pad + y, pad + x, c] = np.median(tmp[y:y + K_size, x:x + K_size, c])

    img2 = np.clip(img2, 0, 255)
    img2 = img2[pad:pad + H, pad:pad + W]

    return img2


img = cv2.imread("assert/imori_noise.jpg").astype(np.float32)
img2 = median_filter(img, 3).astype(np.uint8)
cv2.imshow("imori", img2)
cv2.waitKey(0)
