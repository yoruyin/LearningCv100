import cv2
import numpy as np

def BGR2Grayscale(img):
    r = img[:, :, 2].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 0].copy()

    img2 = 0.2126*r + 0.7152*g + 0.0722*b

    return img2

def Emboss_filter(img, K_size=3):
    H, W = img.shape

    # zero padding
    pad = K_size // 2
    img2 = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
    img2[pad:pad + H, pad:pad + W] = img.copy().astype(np.float)

    tmp = img2.copy()

    K = [[-2.,-1.,0.],
         [-1.,1.,1.],
         [0.,1.,2.]]

    # filterling
    for y in range(H):
        for x in range(W):
                img2[pad + y, pad + x ] = np.sum(K*tmp[y:y + K_size, x:x + K_size ])
    img2 = np.clip(img2,0,255)
    img2 = img2[pad:pad + H, pad:pad + W]

    return img2


img = cv2.imread("assert/imori.jpg")
img2 = BGR2Grayscale(img)
img2 = Emboss_filter(img2, 3).astype(np.uint8)
cv2.imshow("imori", img2)
cv2.waitKey(0)