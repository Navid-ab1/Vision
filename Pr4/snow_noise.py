import cv2
import numpy as np

I = cv2.imread('isfahan.jpg', cv2.IMREAD_GRAYSCALE)
I = I.astype(np.float64) / 255

sigma = 0.04  # initial standard deviation of noise

while True:

    N = np.random.randn(*I.shape)*sigma
    J = I + N
    cv2.imshow('snow noise', J)


    key = cv2.waitKey(33)
    if key & 0xFF == ord('u') and (sigma+0.2)<=1:
        sigma+=0.2
    elif key & 0xFF == ord('d') and (sigma-0.2)>=0:
        sigma-=0.2
    elif key & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
