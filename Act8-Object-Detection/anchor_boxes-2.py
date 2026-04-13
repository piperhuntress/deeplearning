import cv2
import tensorflow as tf
import numpy as np

# Load the image
image = cv2.imread('images/catdog.png')
image_display = image.copy()  # copy to draw on

# Ground truth box (x_min, y_min, x_max, y_max)
ground_truth_box = tf.constant([50, 50, 200, 180], dtype=tf.float32)

# Three anchor boxes (initial guesses)
anchor_boxes = tf.constant([
    [30.0, 40.0, 180.0, 160.0],  # anchor 1
    [40.0, 30.0, 170.0, 150.0],  # anchor 2
    [20.0, 50.0, 190.0, 170.0]   # anchor 3
], dtype=tf.float32)

# Function to compute IoU
def compute_iou(boxes1, box2):
    x1 = tf.maximum(boxes1[:, 0], box2[0])
    y1 = tf.maximum(boxes1[:, 1], box2[1])
    x2 = tf.minimum(boxes1[:, 2], box2[2])
    y2 = tf.minimum(boxes1[:, 3], box2[3])
    
    intersection = tf.maximum(0., x2 - x1) * tf.maximum(0., y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    iou = intersection / union
    return iou

# Compute IoU for each anchor box
ious = compute_iou(anchor_boxes, ground_truth_box)
print("IoUs with ground truth:", ious.numpy())

# Choose the anchor box with the highest IoU
best_anchor_idx = tf.argmax(ious)
print("Best anchor box index:", best_anchor_idx.numpy())
print("Best anchor box coordinates:", anchor_boxes[best_anchor_idx].numpy())

# Draw ground truth box in green
gt = ground_truth_box.numpy().astype(int)
cv2.rectangle(image_display, (gt[0], gt[1]), (gt[2], gt[3]), (0, 255, 0), 2)
cv2.putText(image_display, "GT", (gt[0], gt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

# Draw anchor boxes: blue for normal, red for best
for i, box in enumerate(anchor_boxes.numpy()):
    box_int = box.astype(int)
    color = (255, 0, 0)  # blue
    if i == best_anchor_idx:
        color = (0, 0, 255)  # red for best
    cv2.rectangle(image_display, (box_int[0], box_int[1]), (box_int[2], box_int[3]), color, 2)
    cv2.putText(image_display, f"A{i+1}", (box_int[0], box_int[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Show the image
cv2.imshow("Anchor Boxes Example - CatDog", image_display)
cv2.waitKey(0)
cv2.destroyAllWindows()