import cv2
import numpy as np

# Load the image
I = cv2.imread('coins.jpg')

# Convert the image to grayscale
G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
G = cv2.GaussianBlur(G, (5, 5), 0)

# Define parameters for circle detection
canny_high_threshold = 160
min_votes = 30  # minimum no. of votes to be considered as a circle
min_centre_distance = 40

# Detect circles using Hough Circle Transform
circles = cv2.HoughCircles(G, cv2.HOUGH_GRADIENT, dp=1, minDist=min_centre_distance,
                           param1=canny_high_threshold, param2=min_votes, minRadius=10, maxRadius=100)

# Check if circles are found
if circles is not None:
    circles = np.uint16(np.around(circles))
    n = len(circles[0])

    # Draw detected circles
    for c in circles[0, :]:
        x, y, r = c[0], c[1], c[2]
        cv2.circle(I, (x, y), r, (0, 255, 0), 2)

    # Display the number of detected coins
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(I, 'There are %d coins!' % n, (400, 40), font, 1, (255, 0, 0), 2)

    # Show the image with detected circles
    cv2.imshow("I", I)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No circles detected.")
