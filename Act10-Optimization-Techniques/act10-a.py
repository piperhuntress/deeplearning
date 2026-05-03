# Import Libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize (0–255 → 0–1)
x_train = x_train / 255.0
x_test = x_test / 255.0


# Build a Simple Neural Network
def create_model():
    model = models.Sequential(
        [
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    return model


# Apply Regularization (Prevent Overfitting)
def create_model_with_dropout():
    model = models.Sequential(
        [
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    return model


# Compile Model with Different Optimizers
# Experiment with Learning Rate
# Experiment 1: SGD, learning rate = 0.01
# Experiment 2: Adam, learning rate = 0.001
# Experiment 3: RMSProp, learning rate = 0.0005
learning_rate = 0.01
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

model = create_model()  # no regularization
# model = create_model_with_dropout() # with regularization
model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Train the Model
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Visualize Performance
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.legend(loc="lower right")

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Comparison")
plt.legend(loc="upper right")

plt.tight_layout()
plt.show()


# Evaluate Model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)
print("Loss :", test_loss)
