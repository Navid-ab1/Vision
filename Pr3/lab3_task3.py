import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the input image
input_image = cv2.imread("pasargadae.jpg", cv2.IMREAD_GRAYSCALE)
levels = 256

# Calculate histogram
def calculate_histogram(image, levels):
    hist = np.zeros(levels)
    for pixel_value in range(levels):
        hist[pixel_value] = np.sum(image == pixel_value)
    return hist

# Calculate cumulative distribution function (CDF)
def calculate_cdf(hist, levels):
    cdf = np.zeros_like(hist)
    cdf[0] = hist[0]
    for i in range(1, levels):
        cdf[i] = cdf[i-1] + hist[i]
    return cdf

# Normalize CDF
def normalize_cdf(cdf):
    return (cdf - cdf.min()) / (cdf.max() - cdf.min())

# Map intensity values using normalized CDF
def map_intensity(cdf_normalized, levels):
    mapping = (cdf_normalized * (levels - 1)).astype(np.uint8)
    return mapping

# Replace intensity values in the input image
def replace_intensity(image, mapping):
    equalized_image = mapping[image]
    return equalized_image

# Calculate histogram of the equalized image
def calculate_equalized_histogram(equalized_image, levels):
    hist = np.zeros(levels)
    for pixel_value in range(levels):
        hist[pixel_value] = np.sum(equalized_image == pixel_value)
    return hist

# Calculate CDF of the equalized image
def calculate_equalized_cdf(equalized_image_hist, levels):
    cdf = np.zeros_like(equalized_image_hist)
    cdf[0] = equalized_image_hist[0]
    for i in range(1, levels):
        cdf[i] = cdf[i-1] + equalized_image_hist[i]
    return cdf

# Calculate histogram of the input image
histogram_input = calculate_histogram(input_image, levels)

# Calculate CDF of the input image
cdf_input = calculate_cdf(histogram_input, levels)

# Normalize the CDF
cdf_normalized_input = normalize_cdf(cdf_input)

# Map intensity values using normalized CDF
mapped_intensity = map_intensity(cdf_normalized_input, levels)

# Replace intensity values in the input image
equalized_image = replace_intensity(input_image, mapped_intensity)

# Calculate histogram of the equalized image
equalized_histogram = calculate_equalized_histogram(equalized_image, levels)

# Calculate CDF of the equalized image
cdf_equalized_image = calculate_equalized_cdf(equalized_histogram, levels)

# Display results
fig = plt.figure(figsize=(16, 8))

fig.add_subplot(2, 3, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

fig.add_subplot(2, 3, 2)
plt.plot(histogram_input)
plt.title('Original Histogram')

fig.add_subplot(2, 3, 3)
plt.plot(cdf_input)
plt.title('Original CDF')

fig.add_subplot(2, 3, 4)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

fig.add_subplot(2, 3, 5)
plt.plot(equalized_histogram)
plt.title('Equalized Histogram')

fig.add_subplot(2, 3, 6)
plt.plot(cdf_equalized_image)
plt.title('Equalized CDF')

plt.tight_layout()
plt.show()
