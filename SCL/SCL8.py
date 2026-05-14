import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
(images_train, labels_train), (images_test, labels_test) = (
    tf.keras.datasets.mnist.load_data()
)
# Display dataset shapes
print("Training Images:", images_train.shape)
print("Testing Images:", images_test.shape)
print("testing labels:", labels_test.shape)
images_train = images_train / 255.0
images_test = images_test / 255.0

# Display first 5 training images
plt.figure(figsize=(10, 2))  # 10-width, 2=height in inches
size = len(images_train)


# plt.subplot(1, 2, 1)
# #display 1st image
# plt.imshow(images_train[0])

# #last image
# plt.subplot(1, 2, 2)
# plt.imshow(images_train[size-1])
# plt.show()

# Create a subplot

# using a loop

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(images_train[i])

plt.show()

# Create a basic NN
model = models.Sequential()
model.add(layers.Input(shape=(28, 28, 1)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(1024, activation="relu"))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Train the model
history = model.fit(
    images_train, labels_train, epochs=5, validation_data=(images_test, labels_test)
)

# print(history.history.keys())

# Display Training Results Graph - Accuracy

plt.figure(figsize=(8, 4))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Evaluate the
test_loss, test_accuracy = model.evaluate(images_test, labels_test)

print("Test Accuracy:", test_accuracy)

predictions = model.predict(images_test)
print(predictions[0])

# Predict first test image
predicted_label = np.argmax(predictions[5])


print("Predicted Digit:", predicted_label)
print("Actual Digit:", labels_test[5])


# Display Prediction Result
plt.imshow(images_test[5], cmap="gray")
plt.title(f"Predicted Digit: {predicted_label}")
plt.axis("off")
plt.show()
