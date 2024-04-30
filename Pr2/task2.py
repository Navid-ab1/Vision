import cv2
import numpy as np

im1 = cv2.imread('damavand.jpg')
im2 = cv2.imread('eram.jpg')
for i in np.arange(1.0, 0.0, -0.1):
    k = cv2.addWeighted( im1,i , im2, 1-i,0)
    cv2.imshow('Blending', k)
    cv2.waitKey(500)
cv2.destroyAllWindows()
cv2.destroyAllWindows()
