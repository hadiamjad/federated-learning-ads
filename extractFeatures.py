import cv2

# Load the image
img = cv2.imread("screenshots/1461270.webp")

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create an ORB object
orb = cv2.ORB_create()

# Detect keypoints in the image
keypoints = orb.detect(gray_img, None)

# List the detected keypoints
for kp in keypoints:
    print("Keypoint: x={}, y={}, size={}, angle={}".format(kp.pt[0], kp.pt[1], kp.size, kp.angle))

# Draw keypoints on the original image
img_with_keypoints = cv2.drawKeypoints(img, keypoints, None)

# Display the original image with keypoints
cv2.imshow("Advertisement Image with Keypoints", img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
