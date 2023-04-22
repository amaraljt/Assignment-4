import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input
from tensorflow.keras import Model

# Load text file of nursery rhymes
with open("tmp/nursery_rhymes.txt", "r") as f:
    text = f.read()

# Convert characters to integers
chars = sorted(list(set(text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
num_chars = len(chars)
print("Total characters:", num_chars)

# Prepare the input and output data for the LSTM
seq_length = 20
data = []
labels = []
for i in range(0, len(text) - seq_length, 1):
    seq_in = text[i:i + seq_length]
    seq_out = text[i + seq_length]
    data.append([char_to_int[char] for char in seq_in])
    labels.append(char_to_int[seq_out])

# Reshape the input data to have shape (num_sequences, seq_length, 1)
num_sequences = len(data)
data = np.reshape(data, (num_sequences, seq_length, 1))

# Reshape the output labels to be compatible with categorical_crossentropy loss
labels = np.eye(num_chars)[labels]
labels = np.reshape(labels, (num_sequences, num_chars))

# Define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(seq_length, 1)))
model.add(Dense(num_chars, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(data, labels, epochs=50, batch_size=64)

# Generate new text using the trained model
start_index = np.random.randint(0, len(data) - seq_length - 1)
seed_text = text[start_index:start_index + seq_length]
generated_text = seed_text
for i in range(30 * seq_length):
    x = np.reshape([char_to_int[char] for char in seed_text], (1, seq_length, 1))
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    generated_text += result
    seed_text = seed_text[1:] + result

print(generated_text)