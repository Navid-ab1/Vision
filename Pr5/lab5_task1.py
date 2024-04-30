import cv2
import numpy as np

cam_id = 0  # camera id

cap = cv2.VideoCapture(cam_id)


mode = 'o'
sigma = 5

while True:
    ret, I = cap.read()
    # I = cv2.imread("agha-bozorg.jpg")
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    Ib = cv2.GaussianBlur(I, (sigma, sigma), 0)  # blur the image

    if mode == 'o':
        J = I
    elif mode == 'x':
        J = np.abs(cv2.Sobel(Ib, cv2.CV_64F, 1, 0))

    elif mode == 'y':
        J = np.abs(cv2.Sobel(Ib, cv2.CV_64F, 0, 1))

    elif mode == 'm':
        J_x = np.abs(cv2.Sobel(Ib, cv2.CV_64F, 1, 0))
        J_y = np.abs(cv2.Sobel(Ib, cv2.CV_64F, 0, 1))
        J = np.sqrt(J_x ** 2 + J_y ** 2)

    elif mode == 's':
        threshold_value = 100
        J_x = np.abs(cv2.Sobel(Ib, cv2.CV_64F, 1, 0))
        J_y = np.abs(cv2.Sobel(Ib, cv2.CV_64F, 0, 1))
        r = np.sqrt(J_x ** 2 + J_y ** 2)
        J = (r > threshold_value).astype(np.uint8) * 255


    elif mode == 'l':
        J = cv2.Laplacian(Ib, cv2.CV_64F)


    elif mode == 'c':
        J = cv2.Canny(Ib, 10, 70)

    J = J.astype(np.float64) / J.max()
    cv2.imshow("my stream", J)

    key = chr(cv2.waitKey(1) & 0xFF)

    if key in ['o', 'x', 'y', 'm', 's', 'c', 'l']:
        mode = key
    if key == '-' and sigma > 1:
        sigma -= 2
        print("sigma = %d" % sigma)
    if key in ['+', '=']:
        sigma += 2
        print("sigma = %d" % sigma)
    elif key == 'q':
        break

cap.release()
cv2.destroyAllWindows()
