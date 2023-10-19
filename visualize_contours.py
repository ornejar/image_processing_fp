import cv2
import matplotlib.pyplot as plt

# Read the original image and find contours
original_image = cv2.imread("dent1.jpg", cv2.IMREAD_GRAYSCALE)
contours, _ = cv2.findContours(original_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract x, y coordinates of contour points
contour_points = []
for contour in contours:
    for point in contour:
        x, y = point[0]
        contour_points.append((x, y))

# Unzip the points for scatter plot
x_points, y_points = zip(*contour_points)

# Create scatter plot for contour points
plt.figure(figsize=(12, 6))

# Plot original image
plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title("Original Image")

# Plot contour points
plt.subplot(1, 2, 2)
plt.scatter(x_points, y_points, c='blue', marker='.', linewidths=0.5)
plt.title("Contour Points")
plt.xlabel("X")
plt.ylabel("Y")

plt.tight_layout()
plt.show()
