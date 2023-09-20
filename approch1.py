import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Preprocesses the input image for anomaly detection.

    Args:
        image_path (str): Path to the input image.

    Returns:
        np.ndarray: Preprocessed image (grayscale, resized, etc.).
    """
    try:
        # Load the image
        image = cv2.imread(image_path)

        # Grayscale conversion
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the image (optional, adjust dimensions as needed)
        resized_image = cv2.resize(grayscale_image, (800, 600))  # Example dimensions

        return resized_image

    except Exception as e:
        print(f"Preprocessing error: {str(e)}")
        return None

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

        # Apply a threshold to highlight anomalies
        _, thresholded = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Find contours of anomalies
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around anomalies
        result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        return result_image

    except Exception as e:
        print(f"Anomaly detection error: {str(e)}")
        return None

if __name__ == "__main__":
    image_path = "forest-wf.jpg"
    
    # Preprocess the input image
    preprocessed_image = preprocess_image(image_path)

    if preprocessed_image is not None:
        # Detect anomalies using the preprocessed image
        detected_image = anomaly_detection(preprocessed_image)

        if detected_image is not None:
            # Display the result with detected anomalies
            cv2.imshow("Anomaly Detection", detected_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
