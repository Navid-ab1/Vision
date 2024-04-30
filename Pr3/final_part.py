import cv2
import matplotlib.pyplot as plt
import numpy as np

def expand_histogram(image):
    a = np.percentile(image, 1)
    b = np.percentile(image, 99)
    expanded_image = (image - a) * 255.0 / (b - a)
    expanded_image = np.clip(expanded_image, 0, 255).astype(np.uint8)
    return expanded_image

def plot_histograms(image, expanded_image):
    f, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(image, 'gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[1, 0].hist(image.ravel(), 256, (0, 256), color='red')
    axes[1, 0].set_title('Original Histogram')

    axes[0, 1].imshow(expanded_image, 'gray')
    axes[0, 1].set_title('Expanded Image')
    axes[0, 1].axis('off')

    axes[1, 1].hist(expanded_image.ravel(), 256, (0, 256), color='green')
    axes[1, 1].set_title('Histogram of Expanded Image')

    plt.tight_layout()
    plt.show()

# Example usage:
def process_image(image_name):
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    expanded_image = expand_histogram(image)
    plot_histograms(image, expanded_image)

image_names = ['crayfish.jpg', 'map.jpg', 'train.jpg', 'branches.jpg', 'terrain.jpg']
for name in image_names:
    process_image(name)
