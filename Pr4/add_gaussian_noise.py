import cv2
import numpy as np

I = cv2.imread('isfahan.jpg',cv2.IMREAD_GRAYSCALE)
I = I.astype(np.float64) / 255

sigma = 0.9  # notice maximum intensity is 1
#increase sigma increased noise intensity

N = np.random.randn(*I.shape) * sigma

# add noise to the original image
J = I + N  # or use cv2.add(I,N);

cv2.imshow('original', I)
cv2.waitKey(0)  # press any key to exit

cv2.imshow('noisy image', J)
cv2.waitKey(0)  # press any key to exit

cv2.destroyAllWindows()
