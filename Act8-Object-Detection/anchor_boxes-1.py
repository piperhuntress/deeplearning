import cv2
import tensorflow as tf
import numpy as np


# Load the image
image = cv2.imread('images/catdog.png')
image_display = image.copy()  # copy to draw on

# Ground truth box (x_min, y_min, x_max, y_max) – example: dog
ground_truth_box = tf.constant([50, 50, 200, 180], dtype=tf.float32)

# One anchor box (initial guess)
anchor_box = tf.Variable([30.0, 40.0, 180.0, 160.0])  # just one guess

# Function to compute IoU
def compute_iou(box1, box2):
    x1 = tf.maximum(box1[0], box2[0])
    y1 = tf.maximum(box1[1], box2[1])
    x2 = tf.minimum(box1[2], box2[2])
    y2 = tf.minimum(box1[3], box2[3])
    
    intersection = tf.maximum(0., x2 - x1) * tf.maximum(0., y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    iou = intersection / union
    return iou

# Compute IoU
iou = compute_iou(anchor_box, ground_truth_box)
print("Initial IoU:", iou.numpy())

# Draw ground truth box in green
gt = ground_truth_box.numpy().astype(int)
cv2.rectangle(image_display, (gt[0], gt[1]), (gt[2], gt[3]), (0, 255, 0), 2)
cv2.putText(image_display, "GT", (gt[0], gt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

# Draw anchor box in blue
ab = anchor_box.numpy().astype(int)
cv2.rectangle(image_display, (ab[0], ab[1]), (ab[2], ab[3]), (255, 0, 0), 2)
cv2.putText(image_display, "Anchor", (ab[0], ab[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

# Show the image
cv2.imshow("Anchor Box Example - CatDog", image_display)
cv2.waitKey(0)
cv2.destroyAllWindows()