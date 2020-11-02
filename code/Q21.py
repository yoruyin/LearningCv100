import cv2
import numpy as np
import matplotlib.pyplot as plt

def hist_normalization(img,a = 0,b = 255):
    c = img.min()
    d = img.max()
    out = img.copy()
    out = (b-a)/(d-c)*(out-c)+a
    out[out < a] = a
    out[out > b] = b
    out = out.astype(np.uint8)
    return out


img = cv2.imread("assert/imori_dark.jpg").astype(np.float)
H, W, C = img.shape

out = hist_normalization(img)

plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.show()

cv2.imshow("",out)
cv2.waitKey(0)