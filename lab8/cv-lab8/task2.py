import cv2
import numpy as np

I = cv2.imread('sign.jpg')

p1 = (135, 105)
p2 = (331, 143)
p3 = (356, 292)
p4 = (136, 290)

points1 = np.array([p1, p2, p3, p4], dtype=np.float32)

w = 480
h = 320
output_size = (w, h)

#
# w = 240  # width of the extracted plate
# h = 240  # height of the extracted plate
#
# w = 160  # width of the extracted plate
# h = 240  # height of the extracted plate


points2 = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

H = cv2.getPerspectiveTransform(points1, points2)


J = cv2.warpPerspective(I, H, output_size)

# Display the result
cv2.imshow('J', J)
cv2.waitKey(0)
cv2.destroyAllWindows()
