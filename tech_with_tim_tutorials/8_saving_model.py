import tensorflow as tf
from tensorflow import keras
import numpy as np

TRAIN_MODEL = False
FILE_PATH = "tech_with_tim_tutorials/models/model.h5"

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

word_index = data.get_word_index()
word_index = {key:(value+3) for key, value in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(
    train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(
    test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

model = None
if TRAIN_MODEL:
    model = keras.Sequential()
    model.add(keras.layers.Embedding(88000, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.summary()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    x_val = train_data[:10000]
    x_train = train_data[10000:]
    y_val = train_labels[:10000]
    y_train = train_labels[10000:]

    fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

    results = model.evaluate(test_data, test_labels)
    print(results)

    model.save(FILE_PATH)
else:
    model = keras.models.load_model(FILE_PATH)

def review_encode(s: str):
    encoded = [1] # starting tag <START>
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2) # <UNK>

    return encoded

with open("tech_with_tim_tutorials/test_files/reviews.txt", encoding="utf-8") as file:
    for line in file.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])