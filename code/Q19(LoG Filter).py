import cv2
import numpy as np

def BGR2Grayscale(img):
    r = img[:, :, 2].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 0].copy()

    img2 = 0.2126*r + 0.7152*g + 0.0722*b

    return img2

def LoG_filter(img, K_size=5, sigma=3):
    H, W = img.shape

    ## Zero padding
    pad = K_size // 2
    img2 = np.zeros((H + pad * 2, W + pad * 2 ), dtype=np.float)
    img2[pad:pad + H, pad:pad + W] = img.copy().astype(np.float)

    ## prepare Kernel
    K = np.zeros((K_size, K_size), dtype=np.float)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] =(x**2+y**2-sigma**2)* np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * (sigma ** 6))
    K /= K.sum()
    tmp = img2.copy()

    ## filtering
    for y in range(H):
        for x in range(W):
                img2[pad + y, pad + x ] = np.sum(K * tmp[y:y + K_size, x:x + K_size ])

    img2 = np.clip(img2, 0, 255)
    img2 = img2[pad:pad + H, pad:pad + W]

    return img2


img = cv2.imread("assert/imori_noise.jpg").astype(np.float32)
img2 = BGR2Grayscale(img)
img2 = LoG_filter(img2, 5, 3).astype(np.uint8)
cv2.imshow("imori", img2)
cv2.waitKey(0)