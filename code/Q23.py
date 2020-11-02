import cv2
import numpy as np
import matplotlib.pyplot as plt

def hist_equal(img,z_max=255):
    S = H * W * C *1.
    out = img.copy()
    sum_h = 0.
    for i in range(1,255):
        ind = np.where(out == i)
        sum_h += len(img[ind])
        z_prime = z_max/S*sum_h
        out[ind] = z_prime
    out = out.astype(np.uint8)
    return out


img = cv2.imread("assert/imori.jpg").astype(np.float)
H, W, C = img.shape

out = hist_equal(img)

plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.show()

cv2.imshow("",out)
cv2.waitKey(0)