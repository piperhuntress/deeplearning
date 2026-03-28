import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


# ------Function definitions------------#
def classify_image(uploaded_image, model):
    #  Convert image to a format the computer understands (an array)
    x = image.img_to_array(uploaded_image)
    x = np.expand_dims(x, axis=0)  # Add a 'batch' dimension
    x = preprocess_input(x)  # Adjust colors to match what VGG16 expects

    # Make a prediction
    preds = model.predict(
        x
    )  # Asks the trained VGG16 model to guess what is in your image.

    # Convert the math results into human-readable labels
    results = decode_predictions(preds, top=3)[0]

    st.subheader("Classification Results:")
    # Loop through the results to print them cleanly
    for i, (imagenet_id, label, score) in enumerate(results):
        # Score is a decimal (e.g., 0.98), so we multiply by 100 for a percentage
        st.text(f"{i + 1}. {label}: {score * 100:.2f}%")


# Function for uploading file
def upload_file():
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file)
        st.image(uploaded_image, caption="Uploaded Image", width=700)
        return uploaded_file, uploaded_image
    else:
        return None, None


# ------End Function definitions------------#

# App Information
st.title("VGG16-CNN-Classification")
st.header("Developed by: Jasmine")

app_desc = "This Streamlit-based image classification app uses the VGG16 Convolutional Neural Network (CNN) pre-trained on the ImageNet dataset to identify objects in uploaded images. The app accepts JPG/PNG images, resizes them to 224x224 pixels, and preprocesses them using Keras utilitie. Predictions are generated via TensorFlow/Keras, and the top 3 class labels with confidence scores are displayed. The interface is built with Streamlit, utilizing components like file uploaders, columns, and image display for a simple and interactive user experience."

st.write(app_desc)

# Load the model (pre-trained on the ImageNet dataset)
model = VGG16(weights="imagenet")


# Image Upload
st.subheader("Image Upload")
# Call the upload_file function
uploaded_file, uploaded_image = upload_file()
# Resize image to 224x224 pixels, as VGG16 requires

# Show image information
if uploaded_file is not None:
    uploaded_image = uploaded_image.resize((224, 224))
    st.write(f"Image size: {uploaded_image.size} pixels")
    st.write(f"File name: {uploaded_file.name}")
    # Call the classify_image function
    classify_image(uploaded_image, model)
else:
    st.info("No image uploaded yet. Upload an image.")
