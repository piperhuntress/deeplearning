import cv2
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('images/catdog.png')  # replace with your image

# Convert to RGB for display
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define ONE bounding box
# (x_min, y_min, x_max, y_max)
x_min, y_min, x_max, y_max = 50, 50, 200, 200

# Draw the box
cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

# Show image
plt.imshow(image)
plt.title("Single Bounding Box")
plt.axis('off')
plt.show()