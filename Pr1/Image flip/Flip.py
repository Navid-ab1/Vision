import matplotlib.pyplot as plt
import numpy as np

i = plt.imread('masoleh_gray.jpg')
w = np.flip(i, axis=0)
m = np.concatenate((i, w))
plt.imshow(m, cmap='gray')
plt.show()
