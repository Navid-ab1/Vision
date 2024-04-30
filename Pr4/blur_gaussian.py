import cv2
import numpy as np

I = cv2.imread('isfahan.jpg').astype(np.float64) / 255

m = 12
Fg = cv2.getGaussianKernel(m, sigma=-1)
# by setting sigma=-1, the value of sigma is computed 
# automatically as: sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8

print(Fg)
print(Fg.shape)

Fg = Fg.dot(Fg.T)  # an "m by 1" matrix multiplied by a "1 by m" matrix

print(Fg)
print(Fg.shape)


# filter the image with the Gaussian filter
Jg = cv2.filter2D(I, -1, Fg)

cv2.imshow('original', I)
cv2.waitKey()

cv2.imshow('blurred_Gaussian', Jg)
cv2.waitKey()

cv2.destroyAllWindows()
