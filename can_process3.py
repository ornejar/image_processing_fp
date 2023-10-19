#!/usr/bin/env python
import cv2
import numpy as np
from matplotlib import pyplot as plt

#     return binary_image
def preprocess_image(image_path):
    """
    Preprocesses the input image for dent detection.

    Args:
        image_path (str): File path of the input image.

    Returns:
        np.ndarray: Preprocessed binary image with dents highlighted.
    """
    # Read the original image
    original_image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to create a binary image (adjust threshold value as needed)
    _, binary_image = cv2.threshold(grayscale_image, 235, 255, cv2.THRESH_BINARY_INV)
    
    # Apply morphological operations (dilation and erosion) to enhance features
    kernel = np.ones((5,5),np.uint8)
    binary_image = cv2.dilate(binary_image, kernel, iterations=1)
    binary_image = cv2.erode(binary_image, kernel, iterations=1)
    
    # Ensure the data type is np.uint8 (CV_8UC1)
    binary_image = np.uint8(binary_image)
    
    return binary_image


def resize_image(image, target_size=(256, 256)):
    """
    Resizes the input image to the specified target size.

    Args:
        image (np.ndarray): Input image.
        target_size (tuple): Target size for resizing (width, height).

    Returns:
        np.ndarray: Resized image.
    """
    # Resize the image to the target size
    resized_image = cv2.resize(image, target_size)
    return resized_image

def keypoints_homography(image, reference):
    """
    Applies keypoint matching and homography to align the input image with the reference image.

    Args:
        image (np.ndarray): Input image.
        reference (np.ndarray): Reference image.

    Returns:
        np.ndarray: Aligned image.
    """
    # # Convert images to grayscale
    # image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # reference_grey = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    
    # Initiate ORB detector
    orb = cv2.ORB_create(1000)  # Registration works with at least 50 points
    
    # find the keypoints and descriptors with orb
    kp1, des1 = orb.detectAndCompute(image, None)  # kp1 --> list of keypoints
    kp2, des2 = orb.detectAndCompute(reference, None)
    
    # Match descriptors using brute-force Hamming distance
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(des1, des2, None)  # Creates a list of all matches, just like keypoints
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for match in matches:
        if match.distance < 0.75 * min([m.distance for m in matches]):
            good_matches.append(match)
    
    # Extract matching points
    points1 = np.zeros((len(matches), 2), dtype=np.float32)  # Prints empty array of size equal to (matches, 2)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt  # Gives index of the descriptor in the list of query descriptors
        points2[i, :] = kp2[match.trainIdx].pt  # Gives index of the descriptor in the list of train descriptors
    
    # Apply RANSAC-based homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    print("Homography Matrix:")
    print(h)

    # Check if rotation is needed
    if h is None or np.array_equal(h, np.eye(3)):  # np.eye(3) is the identity matrix
        print("Image doesn't need rotation.")
        return None
    
    img_kp1 = cv2.drawKeypoints(image, kp1, None, color=(0, 255, 0), flags=0)
    img_kp2 = cv2.drawKeypoints(reference, kp2, None, color=(0, 255, 0), flags=0)
    img_matches = cv2.drawMatches(image, kp1, reference, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display images with keypoints and matches
    cv2.imshow("Keypoints Image 1", img_kp1)
    cv2.imshow("Keypoints Image 2", img_kp2)
    cv2.imshow("Matches", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Apply homography to align the image
    height, width = reference.shape  # Grayscale images have only height and width
    aligned_image = cv2.warpPerspective(image, h, (width, height)) # Applies a perspective transformation to an image.

    return aligned_image


def find_contours(image):
    """
    Finds contours in the input binary image.

    Args:
        image (np.ndarray): Binary image.

    Returns:
        list: List of contours found in the image.
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def compare_contours(contour1, contour2, image, threshold=0.1):
    # Draw contours on the image for visualization
    cv2.drawContours(image, [contour1], -1, (0, 255, 0), 2)
    cv2.drawContours(image, [contour2], -1, (0, 0, 255), 2)

    # Compare the similarity of two contours using matchShapes
    similarity = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0.0)
    return similarity < threshold



def main():
    """
    Main function to process the input image and detect dents.
    """
    # Specify the path to your image
    image_path = "can1.jpg"  
    reference_image_path = "can2.jpg"

    original_image = cv2.imread(image_path)
    image_reference = cv2.imread(reference_image_path)
    # Preprocess the image
    image_binary_image = preprocess_image(image_path)
    image_resized_image = resize_image(image_binary_image)

    reference_binary_image = preprocess_image(reference_image_path)
    reference_resized_image = resize_image(reference_binary_image)

    # Fix image if rotated
    rotated_image = keypoints_homography(image_resized_image, reference_resized_image)

    image_contours = find_contours(image_resized_image)
    reference_contours = find_contours(reference_resized_image)
    rotated_image_copy = rotated_image.copy() 

    dent_locations = []
    for image_contour in image_contours:
        for reference_contour in reference_contours:
            if compare_contours(image_contour, reference_contour, image=rotated_image_copy, threshold=0.1):
                # If the contour is similar to the reference, it's not a dent
                continue
            # If the contour is different from the reference, consider it a dent and store its bounding rectangle
            x, y, w, h = cv2.boundingRect(image_contour)
            dent_locations.append((x, y, x + w, y + h))

    image_with_dents = rotated_image
    image_with_dents_color = cv2.merge((image_with_dents, image_with_dents, image_with_dents))
    for dent_location in dent_locations:
        cv2.rectangle(image_with_dents_color, (dent_location[0], dent_location[1]), (dent_location[2], dent_location[3]), (0, 0, 255), 2)
        cv2.rectangle(original_image, (dent_location[0], dent_location[1]), (dent_location[2], dent_location[3]), (0, 0, 255), 2)

    # Add text to the image_with_dents based on the presence of dents
    if len(dent_locations) > 0:
        cv2.putText(original_image, "Dents Found", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image_with_dents_color, "Dents Found", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(original_image, "No Dents", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image_with_dents_color, "No Dents", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the processed image with detected dents
    cv2.imshow("Processed Image with Dents", original_image)
    cv2.imshow("Binary", image_binary_image)
    cv2.imshow("Resize", image_resized_image)
    # cv2.imshow("Contour", image_contours)
    # Optionally, you can also display the original processed image for reference
    cv2.imshow("Processed Image", rotated_image)  # Display the rotated image
    cv2.imshow("Processed Image with Dents and Contours", image_with_dents_color)
    cv2.imshow("Referance Image", image_reference)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

