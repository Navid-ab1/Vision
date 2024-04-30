import cv2
import numpy as np

I = cv2.imread('isfahan.jpg').astype(np.float64) / 255

cv2.imshow('original', I)
cv2.waitKey()

m = 1

# create an m by m box filter
F = np.ones((m, m), np.float64) / (m * m)
print(F)

# Now, filter the image
J = cv2.filter2D(I, -1, F)
cv2.imshow('blurred', J)
cv2.waitKey()

cv2.destroyAllWindows()
