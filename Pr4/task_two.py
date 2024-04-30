import cv2
import numpy as np

I = cv2.imread('isfahan.jpg').astype(np.float64) / 255
j = cv2.blur(I,(8,8))
cv2.imshow('blureed', j)
cv2.waitKey()
cv2.destroyAllWindows()
