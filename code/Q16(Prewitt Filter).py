import cv2
import numpy as np


def BGR2Grayscale(img):
    r = img[:, :, 2].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 0].copy()

    img2 = 0.2126 * r + 0.7152 * g + 0.0722 * b

    return img2


def Prewitt_filter(img, K_size=3):
    H, W = img.shape

    # zero padding
    pad = K_size // 2
    img2 = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
    img2[pad:pad + H, pad:pad + W] = img.copy().astype(np.float)
    tmp = img2.copy()

    out_v = img2.copy()
    out_h = img2.copy()

    Kv = [[-1., -1., -1.], [0., 0., 0.], [1., 1., 1.]]
    Kh = [[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]]

    for y in range(H):
        for x in range(W):
            out_v[pad+y,pad+x] = np.sum(Kv*(tmp[y:y+K_size,x:x+K_size]))
            out_h[pad+y,pad+x] = np.sum(Kh*(tmp[y:y+K_size,x:x+K_size]))

    out_v = np.clip(out_v,0,255)
    out_h = np.clip(out_h,0,255)

    out_v = out_v[pad:pad+H,pad:pad+W].astype(np.uint8)
    out_h = out_h[pad:pad+H,pad:pad+W].astype(np.uint8)

    return out_v,out_h

img = cv2.imread("assert/imori.jpg")
img2 = BGR2Grayscale(img)
img2v ,img2h = Prewitt_filter(img2, 3)
cv2.imshow("imoriv", img2v)
cv2.imshow("imorih", img2h)
cv2.waitKey(0)