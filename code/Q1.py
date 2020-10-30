import cv2

def BGR2RGB(img):
    r = img[:, :, 2].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 0].copy()

    img2 = img.copy()
    img2[:, :, 0] = r
    img2[:, :, 1] = g
    img2[:, :, 2] = b

    return img2


img = cv2.imread("assert/imori.jpg")
img = BGR2RGB(img)
cv2.namedWindow("imori",cv2.WINDOW_AUTOSIZE)
cv2.imshow("imori",img)
cv2.waitKey(0)
cv2.destroyWindow()

