import cv2
import numpy as np

def analyze_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Lower the contour area threshold for smaller dents
        if len(approx) > 5 and cv2.contourArea(contour) > image.shape[0] * image.shape[1] / 20:
            cv2.drawContours(image, [approx], 0, (255, 127, 0), 2)
            message = "Dents"
            cv2.putText(image, message, (image.shape[1] // 5, image.shape[0] // 2), cv2.FONT_HERSHEY_PLAIN, 2.5, (255, 255, 255), 2)
            cv2.imshow("image", image)
            cv2.waitKey()
            return True

    return False

def main():
    # Check if the input image is provided
    # if len(sys.argv) != 2:
    #     print("Usage: python <Sourceprogram> <ImageToLoad>")
    #     return -1

    # Read the input image
    # image_path = sys.argv[1]
    image_path = "pipe3.jpg"
    image = cv2.imread(image_path)

    # Check if the image is loaded successfully
    if image is None:
        print("Could not open or find the image")
        return -1

    # Perform image analysis
    if not analyze_image(image):
        print("Could not find any result")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    main()
