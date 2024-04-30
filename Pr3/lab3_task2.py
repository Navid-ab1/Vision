import cv2
import matplotlib.pyplot as plt
import numpy as np

def hist_operation(name):
    img = cv2.imread(name, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    min_val, max_val = np.min(img_gray), np.max(img_gray)
    final_img = (img_gray - min_val) * 255.0 / (max_val - min_val)
    final_img[final_img < 0] = 0
    final_img[final_img > 255] = 255
    final_img = final_img.astype(np.uint8)

    f, axes = plt.subplots(2, 3, figsize=(12, 12))

    axes[0, 0].imshow(img, 'gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[1, 0].hist(img_gray.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    axes[1, 0].set_title('Original Histogram')

    axes[0, 1].imshow(final_img, cmap='gray')
    axes[0, 1].set_title('Expanded Image')
    axes[0, 1].axis('off')

    axes[1, 1].hist(final_img.ravel(), bins=256, range=(0, 256), color='green', alpha=0.7)
    axes[1, 1].set_title('Histogram of Expanded Image')

    k = cv2.equalizeHist(img_gray)
    axes[0, 2].imshow(k, cmap='gray')
    axes[0, 2].set_title('Equalized Image')
    axes[1, 2].hist(k.ravel(), bins=256, range=(0, 256), color='red', alpha=0.7)
    axes[1, 2].set_title('Histogram Equalized')

    plt.tight_layout()
    plt.show()

hist_operation('crayfish.jpg')
hist_operation('map.jpg')
hist_operation('train.jpg')
hist_operation('branches.jpg')
hist_operation('terrain.jpg')
# fname = 'crayfish.jpg'
# # fname = 'office.jpg'
# img = cv2.imread(fname, cv2.IMREAD_COLOR)
#
# r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
# hist_r = np.zeros(256)
# hist_g = np.zeros(256)
# hist_b = np.zeros(256)
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         hist_r[r[i, j]] += 1
#         hist_g[r[i, j]] += 1
#         hist_b[r[i, j]] += 1
#
# min_r, max_r = np.min(r), np.max(r)
# min_g, max_g = np.min(g), np.max(g)
# min_b, max_b = np.min(b), np.max(b)
#
# re_stretch = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
# gr_stretch = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
# bl_stretch = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
#
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         re_stretch[i, j] = int((r[i, j] - min_r) * 255 / (max_r - min_r))
#         gr_stretch[i, j] = int((g[i, j] - min_g) * 255 / (max_g - min_g))
#         bl_stretch[i, j] = int((b[i, j] - min_b) * 255 / (max_b - min_b))
#
# final_img = cv2.merge((re_stretch, gr_stretch, bl_stretch))
#
# f, axes = plt.subplots(2, 3)
#
# axes[0, 0].imshow(img, 'gray', vmin=0, vmax=255)
# axes[0, 0].axis('off')
#
# axes[1, 1].imshow(final_img, 'gray', vmin=0, vmax=255)
# axes[1, 1].axis('off')
#
# axes[1, 0].hist(img.ravel(), 256, [0, 256])
# axes[0, 1].hist(final_img.ravel(), 256, [0, 256])
#
# gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# k = cv2.equalizeHist(gray_image)
#
# axes[0, 2].imshow(k, 'gray', vmin=0, vmax=255)
# axes[0, 2].axis('off')
#
# axes[1, 2].hist(k.ravel(), 256, [0, 256])
#
# plt.show()
