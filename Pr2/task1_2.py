import cv2
import numpy as np

i = cv2.imread('damavand.jpg')
j = cv2.imread('eram.jpg')
# print(i.shape)
# print(j.shape)
# k = i.copy()
# k[::2, ::2, :] = j[::2, ::2, :]
# cv2.imshow("Image 1",i)
# cv2.imshow("Image 2",j)
# cv2.imshow("Blending", k)
# cv2.waitKey()
# cv2.destroyAllWindows()


#------------------------------Second method-----------------
k = i//2+j//2
cv2.imshow("Blending",k)
cv2.waitKey(0)
cv2.destroyAllWindows()
k = np.clip((0.8*i+0.2*j),0,255).astype(np.uint8)
cv2.imshow("Blending",k)
cv2.waitKey()

print((0.8*i).dtype)
