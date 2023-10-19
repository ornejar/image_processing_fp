import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the original image
original_image = cv2.imread("dent1.jpg", cv2.IMREAD_GRAYSCALE)

# Normalize pixel intensities to the range [0, 255]
normalized_image = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX)

# Apply thresholding to create a binary image
_, binary_image = cv2.threshold(normalized_image, 235, 255, cv2.THRESH_BINARY_INV)

# Debug prints to check the state of binary_image
print("Type of binary_image:", type(binary_image))
print("Shape of binary_image:", binary_image.shape if binary_image is not None else "None")

# Add an assertion to check if binary_image is not None before performing dilation
assert binary_image is not None, "Binary image is None, cannot perform dilation operation."

# Apply morphological operations (dilation and erosion)
kernel = np.ones((5, 5), np.uint8)
dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
eroded_image = cv2.erode(dilated_image, kernel, iterations=1)

# Plot histograms
plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1)
plt.imshow(normalized_image, cmap='gray')
plt.title("Normalized Image")
plt.axis('off')
plt.subplot(2, 3, 2)
plt.hist(normalized_image.ravel(), 256, [0, 256], color='black')
plt.title("Normalized Image Histogram")
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.subplot(2, 3, 4)
plt.imshow(dilated_image, cmap='gray')
plt.title("Dilated Image")
plt.axis('off')
plt.subplot(2, 3, 5)
plt.hist(dilated_image.ravel(), 256, [0, 256], color='black')
plt.title("Dilated Image Histogram")
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.subplot(2, 3, 3)
plt.imshow(binary_image, cmap='gray')
plt.title("Thresholded Image")
plt.axis('off')
plt.subplot(2, 3, 6)
plt.hist(binary_image.ravel(), 256, [0, 256], color='black')
plt.title("Thresholded Image Histogram")
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
