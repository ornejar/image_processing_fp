import cv2
import numpy as np

# def preprocess_image(image_path):
#     # Read the original image
#     original_image = cv2.imread(image_path)
    
#     # Convert the image to grayscale
#     grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
#     # Apply thresholding to create a binary image (adjust threshold value as needed)
#     _, binary_image = cv2.threshold(grayscale_image, 235, 255, cv2.THRESH_BINARY_INV)
#     # binary_image = cv2.adaptiveThreshold(grayscale_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 8)
    
#     return binary_image
def preprocess_image(image_path):
    # Read the original image
    original_image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
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

# def remove_background(image):
#     # Define the lower and upper boundaries of the color range to be kept (BGR format)
#     lower_bound = np.array([0, 0, 0])  # Lower boundary (black)
#     upper_bound = np.array([100, 100, 100])  # Upper boundary (gray)

#     # Create a mask within the color range
#     mask = cv2.inRange(image, lower_bound, upper_bound)

#     # Bitwise AND between the mask and the original image to keep only the can shape
#     can_shape = cv2.bitwise_and(image, image, mask=mask)

#     return can_shape
def remove_background(image):
    # Define the lower and upper boundaries of the color range to be kept (grayscale format)
    lower_bound = 0  # Lower boundary (black)
    upper_bound = 255  # Upper boundary (gray)

    # Create a mask within the grayscale range
    mask = cv2.inRange(image, lower_bound, upper_bound)

    # Bitwise AND between the mask and the original image to keep only the can shape
    can_shape = cv2.bitwise_and(image, image, mask=mask)

    return can_shape

# def find_contours(image):
#     # Find contours in the binary image
#     contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return contours

# def draw_contours(image, contours):
#     # Draw the contours on the image
#     cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
#     return image

def detect_dents_with_contours(image, template):
    # Find contours in the binary image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dent_locations = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y:y+h, x:x+w]
        
        # Ensure the ROI size is not larger than the template size
        roi = cv2.resize(roi, (template.shape[1], template.shape[0]))
        
        # Match the template in the ROI
        result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        
        # Threshold for match confidence (adjust as needed)
        threshold = 0.6
        
        # If the match confidence is above the threshold, consider it a dent
        if result[max_loc[1], max_loc[0]] > threshold:
            dent_locations.append((x, y, x + w, y + h))
    
    return dent_locations

# def detect_dents(image, template):
#     # Match the template in the image
#     result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
#     _, _, _, max_loc = cv2.minMaxLoc(result)

#     # Threshold for match confidence (adjust as needed)
#     threshold = 0.8

#     # If the match confidence is above the threshold, consider it a dent
#     if result[max_loc[1], max_loc[0]] > threshold:
#         return True, max_loc
#     else:
#         return False, None
def detect_dents(image, template):
    # Match the template in the image
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    
    # Find all the locations where the template matches above a certain threshold
    threshold = 0.9  # Adjust this threshold value if needed
    loc = np.where(result >= threshold)
    
    dent_locations = []
    for pt in zip(*loc[::-1]):
        dent_locations.append((pt[0], pt[1], pt[0] + template.shape[1], pt[1] + template.shape[0]))

    return dent_locations
# def main():
#     # Specify the path to your image
#     image_path = "can1.jpg"
    
#     # Preprocess the image
#     binary_image = preprocess_image(image_path)
#     resized_image = resize_image(binary_image)
#     can_shape = remove_background(resized_image)

#     # Display original image and processed image side by side
#     original_image = cv2.imread(image_path)
#     combined_image = np.hstack((original_image, can_shape))
#     cv2.imshow("Debug: Original vs Processed", combined_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
def anomaly_detection(image):
    """
    Detects anomalies in the preprocessed image using morphological operators.

    Args:
        image (np.ndarray): Preprocessed image.

    Returns:
        np.ndarray: Image with detected anomalies (highlighted by bounding boxes).
    """
    try:
        # Apply morphological operations (e.g., erosion and dilation)
        kernel = np.ones((5, 5), np.uint8)  # Example kernel size
        erosion = cv2.erode(image, kernel, iterations=1)
        dilation = cv2.dilate(image, kernel, iterations=1)

        # Calculate absolute difference between original and processed images
        diff = cv2.absdiff(image, dilation)

        # # Apply a threshold to highlight anomalies
        # _, thresholded = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        thresholded = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)

        # Find contours of anomalies
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around anomalies
        result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        min_contour_area = 100 
        for contour in contours:
            if cv2.contourArea(contour) > min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        return result_image
    except Exception as e:
        print(f"Anomaly detection error: {str(e)}")
        return None
def main():
    # Specify the path to your image
    image_path = "can1.jpg"
    
    # Preprocess the image
    binary_image = preprocess_image(image_path)
    resized_image = resize_image(binary_image)
    can_shape = remove_background(resized_image)

    # Read original image as grayscale
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the original image to match the processed image size
    original_image_resized = resize_image(original_image, target_size=resized_image.shape[:2])

    # Display original image and processed image side by side
    combined_image = np.hstack((original_image_resized, can_shape))


    # Find contours in the processed binary image (can_shape)
    # contours = find_contours(can_shape)
    #  # Draw contours on the resized original image
    # image_with_contours = draw_contours(original_image_resized.copy(), contours)

    # print("Image with Contours Shape:", image_with_contours.shape)

     # Load template images (reference images without dents)
    template_image1 = preprocess_image("can1.jpg")  # Provide path to your template images
    # template_image2 = preprocess_image("can2.jpg")

    # # Detect dents in the processed image using template matching
    # is_dent1, dent_location1 = detect_dents(can_shape, template_image1)
    # is_dent2, dent_location2 = detect_dents(can_shape, template_image2)
    # Detect dents in the processed image using template matching
    dent_locations = detect_dents_with_contours(can_shape, template_image1)
    # dent_locations2 = detect_dents_with_contours(can_shape, template_image2)

    # # Draw rectangles around detected dents
    # if is_dent1:
    #     cv2.rectangle(image_with_contours, dent_location1, (dent_location1[0] + template_image1.shape[1], dent_location1[1] + template_image1.shape[0]), (0, 0, 255), 2)
    # if is_dent2:
    #     cv2.rectangle(image_with_contours, dent_location2, (dent_location2[0] + template_image2.shape[1], dent_location2[1] + template_image2.shape[0]), (0, 0, 255), 2)
#  # Draw rectangles around detected dents
#     for dent_location in dent_locations1:
#         cv2.rectangle(image_with_contours, (dent_location[0], dent_location[1]), (dent_location[2], dent_location[3]), (0, 255, 0), 5)
#     for dent_location in dent_locations2:
#         cv2.rectangle(image_with_contours, (dent_location[0], dent_location[1]), (dent_location[2], dent_location[3]), (0, 255, 0), 5)
     # Draw rectangles around detected dents (set color to green)
    image_with_dents = cv2.cvtColor(can_shape, cv2.COLOR_GRAY2BGR)  # Convert to color for drawing
    for dent_location in dent_locations:
        cv2.rectangle(image_with_dents, (dent_location[0], dent_location[1]), (dent_location[2], dent_location[3]), (0, 0, 255), 10)
    # for dent_location in dent_locations2:
    #     cv2.rectangle(image_with_dents, (dent_location[0], dent_location[1]), (dent_location[2], dent_location[3]), (0, 255, 0), 2)
    detected_image = anomaly_detection(can_shape)
    cv2.imshow("Debug: Original vs Processed", combined_image)
    # Display original image with contours
    # cv2.imshow("Image with Contours", image_with_contours)
    # # Display the result
    # cv2.imshow("Dent Detection Result", image_with_contours)
    cv2.imshow("Dent Detection Result", image_with_dents)
    cv2.imshow("anomaly", detected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # print("Dent Locations 1:", dent_locations1)
    # print("Dent Locations 2:", dent_locations2)

if __name__ == "__main__":
    main()
