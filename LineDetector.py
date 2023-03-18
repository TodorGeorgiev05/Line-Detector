import cv2
import numpy as np

def detect_lines(image_path):
    # Load image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # Define the region of interest
    mask = np.zeros_like(edges)
    height, width = mask.shape[:2]
    vertices = np.array([[(0, height), (width // 2, height // 2), (width, height)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, color=255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(masked_edges, rho=2, theta=np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

    # Draw lines on the original image
    line_thickness = 2
    line_color = (0, 255, 0)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, pt1=(x1, y1), pt2=(x2, y2), color=line_color, thickness=line_thickness)

    # Display the final image
    cv2.imshow('Detected Lines', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
detect_lines('ima.png')
