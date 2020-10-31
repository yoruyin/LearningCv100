import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("assert/imori_dark.jpg").astype(np.float32)


plt.hist(img.ravel(),bins=255,rwidth=0.7,range=(0,255))
plt.show()
