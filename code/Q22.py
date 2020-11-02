import cv2
import numpy as np
import matplotlib.pyplot as plt

def hist_mani(img,m0 = 128,s0 = 52):
    m = img.mean()
    s = img.std()
    out = img.copy()
    out = s0/s*(out-m)+m0
    out[out < 0] = 0
    out[out > 255] = 255
    out = out.astype(np.uint8)
    return out


img = cv2.imread("assert/imori_dark.jpg").astype(np.float)
H, W, C = img.shape

out = hist_mani(img)

plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.show()

cv2.imshow("",out)
cv2.waitKey(0)