# filename: streamlit_object_detection.py

import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from PIL import Image

# -----------------------------
# 1️⃣ Load SSD MobileNet V2 model (cached for performance)
# -----------------------------
@st.cache_resource
def load_model():
    return hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

model = load_model()
st.title("🖼️ Object Detection with SSD MobileNet V2")

# -----------------------------
# 2️⃣ COCO class labels (subset + full)
# -----------------------------
coco_labels = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
    44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife',
    50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich',
    55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza',
    60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant',
    65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop',
    74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave',
    79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
    90: 'toothbrush'
}

# -----------------------------
# 3️⃣ Upload an image
# -----------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    # Load image with PIL and convert to RGB
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image_np, caption="Uploaded Image", width=700)

    # -----------------------------
    # 4️⃣ Prepare image for model
    # -----------------------------
    input_tensor = tf.convert_to_tensor(image_np, dtype=tf.uint8)
    input_tensor = tf.expand_dims(input_tensor, axis=0)

    # -----------------------------
    # 5️⃣ Run detection
    # -----------------------------
    output = model(input_tensor)
    boxes = output['detection_boxes'][0].numpy()
    scores = output['detection_scores'][0].numpy()
    classes = output['detection_classes'][0].numpy()

    # -----------------------------
    # 6️⃣ Draw bounding boxes
    # -----------------------------
    image_draw = image_np.copy()
    h, w, _ = image_draw.shape

    for i in range(len(boxes)):
        if scores[i] >= 0.5:  # confidence threshold
            class_id = int(classes[i])
            class_name = coco_labels.get(class_id, 'N/A')
            ymin, xmin, ymax, xmax = boxes[i]
            left, top, right, bottom = int(xmin*w), int(ymin*h), int(xmax*w), int(ymax*h)

            cv2.rectangle(image_draw, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image_draw, f"{class_name}: {scores[i]:.2f}", 
                        (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    st.image(image_draw, caption="Detected Objects", width=700)