import cv2
import numpy as np

I = cv2.imread('masoleh.jpg')

B = I[:, :, 0]
G = I[:, :, 1]
R = I[:, :, 2]

B = np.zeros(I.shape, dtype='uint8')
B[:, :, 0] = I[:, :, 0]

G = np.zeros(I.shape, dtype='uint8')
G[:, :, 1] = I[:, :, 1]

R = np.zeros(I.shape, dtype='uint8')
R[:, :, 2] = I[:, :, 2]

cv2.imshow('win1', I)

while 1:

    k = cv2.waitKey()

    if k == ord('o'):
        cv2.imshow('win1', I)
    elif k == ord('b'):
        cv2.imshow('win1', B)
    elif k == ord('g'):
        cv2.imshow('win1', G)
    elif k == ord('r'):
        cv2.imshow('win1', R)
    elif k == ord('q'):
        cv2.destroyAllWindows()
        break
