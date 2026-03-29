import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# App Information
st.title("RNN Text Predictor (SimpleRNN / LSTM / GRU)")
st.write("Developed by: Jasmine")

app_desc = "This Streamlit app demonstrates text prediction using Recurrent Neural Networks (RNNs). Users can select between SimpleRNN, LSTM, and GRU models to predict the next word based on an input phrase. The model is trained on a small custom text dataset to showcase sequence learning in deep learning."

st.write(app_desc)

# ----------------------------
# Streamlit User Input
# ----------------------------
user_input = st.text_input("Enter your starting text:", "the cat")

rnn_type = st.selectbox("Choose RNN Type", ["SimpleRNN", "LSTM", "GRU"])


# Sample training data
data = """
the cat sat on the mat
the cat chased the mouse
the cat climbed the tree
the cat slept on the sofa
the cat ate the fish

the dog sat on the rug
the dog chased the cat
the dog barked loudly
the dog ran in the park
the dog ate the food

the mouse ran away
the mouse hid under the table
the mouse ate the cheese
the mouse climbed the wall

the bird flew in the sky
the bird sang a song
the bird sat on the branch
the bird ate the seed

the boy played football
the boy ate an apple
the boy ran to school

the girl read a book
the girl played with the cat
the girl sang a song
"""

# Tokenization
tokenizer = Tokenizer()  # convert words into numbers
tokenizer.fit_on_texts([data])  #
total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
words = data.split()
for i in range(1, len(words)):
    n_gram_sequence = words[: i + 1]
    encoded = tokenizer.texts_to_sequences([n_gram_sequence])[0]
    input_sequences.append(encoded)

# Padding sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")
)

# Split features and labels
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)


# Build model dynamically
def build_model(rnn_type):
    model = Sequential()
    model.add(
        layers.Embedding(
            input_dim=total_words, output_dim=10, input_length=max_sequence_len - 1
        )
    )

    if rnn_type == "SimpleRNN":
        model.add(layers.SimpleRNN(100))
    elif rnn_type == "LSTM":
        model.add(layers.LSTM(100))
    elif rnn_type == "GRU":
        model.add(layers.GRU(100))

    model.add(layers.Dense(total_words, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


model = build_model(rnn_type)
st.write(f"Training {rnn_type} model...")
model.fit(X, y, epochs=200, verbose=0)

# Prediction
if st.button("Predict Next Word"):
    token_list = tokenizer.texts_to_sequences([user_input])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")

    predicted_index = np.argmax(model.predict(token_list), axis=-1)[0]

    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            output_word = word
            break

    st.success(f"Prediction: {user_input} **{output_word}**")
