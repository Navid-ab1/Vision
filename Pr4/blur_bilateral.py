import cv2
import numpy as np

I = cv2.imread('isfahan.jpg').astype(np.float32) / 255
GB = cv2.GaussianBlur(I, (9, 9), 0)
cv2.imshow('original', I)
cv2.waitKey()

cv2.imshow('blurred_Gaussian', GB)
cv2.waitKey()

cv2.destroyAllWindows()


size = 9  # bilateral filter size (diameter)
sigma_color =0.8
sigma_space = 11

Jl = cv2.bilateralFilter(I, size, sigma_color, sigma_space)

cv2.imshow('original', I)
cv2.waitKey()

cv2.imshow('bilateralFilter', Jl)
cv2.waitKey()

cv2.destroyAllWindows()
