import cv2
import numpy as np
import matplotlib.pyplot as plt

def keypoints_homography(image, reference):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    # Initiate ORB detector
    orb = cv2.ORB_create(50)  # Registration works with at least 50 points
    
    # find the keypoints and descriptors with orb
    kp1, des1 = orb.detectAndCompute(image, None)  # kp1 --> list of keypoints
    kp2, des2 = orb.detectAndCompute(reference, None)
    
    # Match descriptors using brute-force Hamming distance
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(des1, des2, None)  # Creates a list of all matches, just like keypoints
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract matching points
    points1 = np.zeros((len(matches), 2), dtype=np.float32)  # Prints empty array of size equal to (matches, 2)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt  # Gives index of the descriptor in the list of query descriptors
        points2[i, :] = kp2[match.trainIdx].pt  # Gives index of the descriptor in the list of train descriptors
    
    # Apply RANSAC-based homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Check if rotation is needed
    if h is None or np.array_equal(h, np.eye(3)):  # np.eye(3) is the identity matrix
        print("Image doesn't need rotation.")
        return None
    
    # Apply homography to align the image
    height, width = reference.shape  # Grayscale images have only height and width
    aligned_image = cv2.warpPerspective(image, h, (width, height))  # Applies a perspective transformation to an image.

    return aligned_image

# Read images
image = cv2.imread("rotate-can1.jpg")
reference = cv2.imread("can1.jpg")

# Obtain registered image using keypoints homography
registered_image = keypoints_homography(image, reference)

# Initiate ORB detector
orb = cv2.ORB_create(50)  # Registration works with at least 50 points

# Find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(image, None)
kp2, des2 = orb.detectAndCompute(reference, None)

# Draw keypoints on the original and registered images
img1_keypoints = cv2.drawKeypoints(image, kp1, None, color=(0, 255, 0), flags=0)
img2_keypoints = cv2.drawKeypoints(registered_image, kp2, None, color=(0, 255, 0), flags=0)

# Display the images with keypoints
plt.figure(figsize=(12, 6))

# Plot keypoints for the original image
plt.subplot(1, 2, 1)
plt.imshow(img2_keypoints)
plt.title("Original Image Keypoints")

# Plot keypoints for the registered image
plt.subplot(1, 2, 2)
plt.imshow(img1_keypoints)
plt.title("Registered Image Keypoints")

plt.tight_layout()
plt.show()
