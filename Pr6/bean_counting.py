import cv2
import numpy as np

I = cv2.imread('beans.jpg')
G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

ret, t = cv2.threshold(G, 127, 255, cv2.THRESH_BINARY)

cv2.imshow('Thresholded', t)
cv2.waitKey(0)  # press any key to continue...



kernel = np.ones((15, 21), np.uint8)
t = cv2.erode(t, kernel)
cv2.imshow('After Erosion',t)
cv2.waitKey(0)  # press any key to continue...

# dilation
kernel = np.ones((10, 10), np.uint8)
t = cv2.dilate(t, kernel)
cv2.imshow('Dilation', t)
cv2.waitKey(0)  # press any key to continue...

n, C = cv2.connectedComponents(t)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(t, 'There are %d beans!' % (n - 1), (20, 40), font, 1, 255, 2)
cv2.imshow('Num', t)
cv2.waitKey(0)
