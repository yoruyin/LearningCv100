import cv2
import numpy as np
import matplotlib.pyplot as plt

def gamma_correction(img,c = 1, g = 2.2):
    out = img.copy()
    out /= 255.
    out = (1/c) * out**(1/g)
    out *= 255
    out = out.astype(np.uint8)
    return out


img = cv2.imread("assert/imori_gamma.jpg").astype(np.float)
H, W, C = img.shape

out = gamma_correction(img)

cv2.imshow("",out)
cv2.waitKey(0)