import cv2
import numpy as np

I = cv2.imread('branches2.jpg').astype(np.float64) / 255

noise_sigma = 0.04  # initial standard deviation of noise

m = 3  # initial filter size,

gm = 3  # gaussian filter size

size = 9  # bilateral filter size
sigmaColor = 0.3
sigmaSpace = 75

# with m = 1 the input image will not change
filter = 'b'  # box filter

while True:

    # add noise to image
    N = np.random.rand(*I.shape) * noise_sigma
    J = I + N
    j_uint8 = np.clip(J * 255, 0, 255).astype(np.uint8)

    if filter == 'b':
        f = np.ones((m, m), np.float64) / (m * m)
        K = cv2.filter2D(j_uint8, -1, f)

    elif filter == 'g':
        K = cv2.GaussianBlur(j_uint8, (gm, gm), 0)

    elif filter == 'l':
        K = cv2.bilateralFilter(j_uint8, size, sigmaColor, sigmaSpace)

    cv2.imshow('img', K)
    key = cv2.waitKey(30) & 0xFF

    if key == ord('b'):
        filter = 'b'  # box filter
        print('Box filter')

    elif key == ord('g'):
        filter = 'g'  # filter with a Gaussian filter
        print('Gaussian filter')

    elif key == ord('l'):
        filter = 'l'  # filter with a bilateral filter
        print('Bilateral filter')

    elif key == ord('+'):
        m = m + 2
        print('m=', m)

    elif key == ord('-'):
        if m >= 3:
            m = m - 2
        print('m=', m)
    elif key == ord('u'):
        if noise_sigma <= 1:
            noise_sigma += 0.1

    elif key == ord('d'):
        if noise_sigma >= 0:
            noise_sigma -= 0.1

    elif key == ord('p'):
        sigmaColor += 0.1

    elif key == ord('n'):
        if sigmaColor >= 0.2:
            sigmaColor -= 0.1

    elif key == ord('>'):
        size = size + 1

    elif key == ord('<'):
        size = size - 1

    elif key == ord('q'):
        break

cv2.destroyAllWindows()
