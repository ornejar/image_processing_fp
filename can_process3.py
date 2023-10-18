#!/usr/bin/env python
import cv2
import numpy as np
from matplotlib import pyplot as plt

#     return binary_image
def preprocess_image(image_path):
    # Read the original image
    original_image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    grayscale_image = np.float32(grayscale_image)
    # Apply thresholding to create a binary image (adjust threshold value as needed)
    _, binary_image = cv2.threshold(grayscale_image, 235, 255, cv2.THRESH_BINARY_INV)
    
    # Apply morphological operations (dilation and erosion)
    kernel = np.ones((5,5),np.uint8)
    binary_image = cv2.dilate(binary_image, kernel, iterations=1)
    binary_image = cv2.erode(binary_image, kernel, iterations=1)
    
    return binary_image

def resize_image(image, target_size=(256, 256)):
    # Resize the image to the target size
    resized_image = cv2.resize(image, target_size)
    return resized_image

def keypoints_homography(image, reference):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    # Initiate ORB detector
    orb = cv2.ORB_create(50)  #Registration works with at least 50 points
    # find the keypoints and descriptors with orb
    kp1, des1 = orb.detectAndCompute(image, None)  #kp1 --> list of keypoints
    kp2, des2 = orb.detectAndCompute(reference, None)
    
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    # Match descriptors.
    matches = matcher.match(des1, des2, None)  #Creates a list of all matches, just like keypoints
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(image,kp1, reference, kp2, matches[:10], None)

    cv2.imshow("Matches image", img3)
    cv2.waitKey(0)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)  #Prints empty array of size equal to (matches, 2)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt    #gives index of the descriptor in the list of query descriptors
        points2[i, :] = kp2[match.trainIdx].pt    #gives index of the descriptor in the list of train descriptors
    
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
    # Use homography
    height, width, channels = reference.shape
    im1Reg = cv2.warpPerspective(image, h, (width, height))  #Applies a perspective transformation to an image.

    return im1Reg

def find_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def compare_contours(contour1, contour2, threshold=0.1):
    # Compare the similarity of two contours using matchShapes
    similarity = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0.0)
    return similarity < threshold



def main():
    # Specify the path to your image
    image_path = "can2.jpg"
    reference_image = "can1.jpg"

    # Preprocess the image
    image_binary_image = preprocess_image(image_path)
    image_resized_image = resize_image(image_binary_image)
    reference_binary_image = preprocess_image(reference_image)
    reference_resized_image = resize_image(reference_binary_image)

    image_contours = find_contours(image_resized_image)
    reference_contours = find_contours(reference_resized_image)

    dent_locations = []
    for image_contour in image_contours:
        for reference_contour in reference_contours:
            if compare_contours(image_contour, reference_contour, threshold=0.1):
                # If the contour is similar to the reference, it's not a dent
                continue
            # If the contour is different from the reference, consider it a dent and store its bounding rectangle
            x, y, w, h = cv2.boundingRect(image_contour)
            dent_locations.append((x, y, x + w, y + h))

    image_with_dents = resize_image(cv2.imread(image_path))
    for dent_location in dent_locations:
        cv2.rectangle(image_with_dents, (dent_location[0], dent_location[1]), (dent_location[2], dent_location[3]), (0, 0, 255), 2)

    # Add text to the image_with_dents based on the presence of dents
    if len(dent_locations) > 0:
        cv2.putText(image_with_dents, "Dents Found", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(image_with_dents, "No Dents", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the processed image with detected dents
    cv2.imshow("Processed Image with Dents", image_with_dents)
    # Optionally, you can also display the original processed image for reference
    cv2.imshow("Processed Image", image_resized_image)
    cv2.imshow("Referance Image", reference_resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
