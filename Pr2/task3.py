import cv2
import numpy as np


def rgb_2_gray(color_img):
    b, g, r = color_img[:, :, 0], color_img[:, :, 1], color_img[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.1140 * b
    gray = gray.astype('uint8')
    return gray


img = cv2.imread('eram.jpg', cv2.IMREAD_COLOR)
gray_img = rgb_2_gray(img)
gray_img_3ch = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
for i in np.arange(1.0, 0.0, -0.1):
    k = cv2.addWeighted(gray_img_3ch, i, img, 1 - i, 0)
    cv2.imshow('Blending', k)
    cv2.waitKey(500)
cv2.destroyAllWindows()
