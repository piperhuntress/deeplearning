import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import os


from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------------
# App Info
# -------------------------------
st.title("Optimized CNN Image Classifier")
st.write("Student Activity: Optimization Techniques")

app_desc = """This Streamlit app allows students to experiment with image classification using a VGG16-based Convolutional Neural Network (CNN). Students can upload images to classify them and explore the effects of different optimization techniques—including Adam, RMSprop, and SGD optimizers, learning rates, dropout, and L2 regularization. The app includes training visualization with accuracy and loss plots, displays final metrics, and allows students to compare optimization settings for hands-on learning in deep learning."""

# -------------------------------
# Sidebar - Training Settings
# -------------------------------
st.sidebar.header("Training Configuration")

optimizer_choice = st.sidebar.selectbox("Choose Optimizer", ["Adam", "RMSprop", "SGD"])

learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001)

epochs = st.sidebar.slider("Epochs", 1, 20, 5)


# -------------------------------
# Load Base Model (VGG16)
# -------------------------------
def build_model(optimizer_name, lr):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False

    # Custom layers with Regularization + Dropout
    x = Flatten()(base_model.output)
    x = Dense(256, activation="relu", kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    output = Dense(3, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    # Optimizer selection
    if optimizer_name == "Adam":
        optimizer = Adam(learning_rate=lr)
    elif optimizer_name == "RMSprop":
        optimizer = RMSprop(learning_rate=lr)
    else:
        optimizer = SGD(learning_rate=lr, momentum=0.9)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# -------------------------------
# Training Section
# -------------------------------
st.header("Model Training")

if st.button("Train Model"):
    st.info("Training started...")

    # Data Generators
    train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_data = train_datagen.flow_from_directory(
        "dataset/",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        subset="training",
    )
    # Save the class names
    class_names = list(train_data.class_indices.keys())
    with open("class_names.json", "w") as f:
        json.dump(class_names, f)
    val_data = train_datagen.flow_from_directory(
        "dataset/",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        subset="validation",
    )

    model = build_model(optimizer_choice, learning_rate)

    # Callbacks
    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    # Train
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[lr_scheduler, early_stop],
    )

    # Save model
    model.save("trained_model.h5")
    st.success("Model trained and saved!")

    # Plot Accuracy
    st.subheader("Training Performance")

    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    st.pyplot(plt)

    # Display final values
    st.write("**Final Metrics:**")
    st.write(f"Train Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
    st.write(f"Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
    st.write(f"Train Loss: {history.history['loss'][-1]:.4f}")
    st.write(f"Validation Loss: {history.history['val_loss'][-1]:.4f}")

# -------------------------------
# Prediction Section
# -------------------------------
st.header("Image Classification")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and resize image
    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img, caption="Uploaded Image", width=300)

    # Load trained model
    try:
        model = load_model("trained_model.h5")
    except:
        st.error("Please train the model first!")
        st.stop()

    # Load class names safely
    if os.path.exists("class_names.json"):
        with open("class_names.json", "r") as f:
            class_names = json.load(f)
    else:
        st.error("Class names file not found. Please train the model first!")
        st.stop()

    # Preprocess image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Predict
    preds = model.predict(x)

    # Get top-3 predictions
    top_indices = preds[0].argsort()[-3:][::-1]
    st.subheader("Top Predictions")
    for i, idx in enumerate(top_indices):
        st.write(f"{i+1}. {class_names[idx]}: {preds[0][idx]*100:.2f}%")
