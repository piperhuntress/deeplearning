import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import streamlit as st
import numpy as np
import pickle
import os

# App Information
st.title("Object Detection with SSD MobileNet V2")
st.header("Developed by: Jasmine")

app_desc = ""

data = "the cat sat on the mat the dog sat on the log the cat chased the dog"
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
total_words = len(tokenizer.word_index) + 1

# Create input sequences (e.g., "the cat" -> "sat")
input_sequences = []
words = data.split()
for i in range(1, len(words)):
    n_gram_sequence = words[: i + 1]
    encoded = tokenizer.texts_to_sequences([n_gram_sequence])[0]
    input_sequences.append(encoded)

# Pad sequences so they are all the same length
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")
)

# Split into X (features) and y (label/target)
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

model = Sequential(
    [
        Embedding(total_words, 10, input_length=max_sequence_len - 1),
        SimpleRNN(100),
        Dense(total_words, activation="softmax"),
    ]
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, y, epochs=200, verbose=0)
# model_path = "text_model.h5"

# # Check if the file exists and try to remove it first
# if os.path.exists(model_path):
#     try:
#         os.remove(model_path)
#     except OSError:
#         print("Warning: Could not delete old model. It might be open in Streamlit.")

# model.save(model_path)
print("Model saved successfully!")

# Load your model (simplified for this activity)
model = tf.keras.models.load_model(model)

st.title("RNN Text Predictor")
st.write("Type a word like 'The cat' and see what the AI thinks comes next!")

user_input = st.text_input("Enter your starting text:", "the cat")

if st.button("Predict Next Word"):
    # 1. Convert input to numbers (Same logic as training)
    token_list = tokenizer.texts_to_sequences([user_input])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")

    # 2. Get prediction
    predicted = np.argmax(model.predict(token_list), axis=-1)

    # 3. Convert number back to word
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break

    st.success(f"Result: {user_input} **{output_word}**")
