from matplotlib import pyplot as plt
import cv2
i = cv2.imread('masoleh.jpg',cv2.IMREAD_UNCHANGED)
# plt.imshow(i)
# plt.show()
i[:,:,1]=0
plt.imshow(i[:,:,::-1])
plt.show()