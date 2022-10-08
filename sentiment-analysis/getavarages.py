import pandas as pd
import numpy as np
import pickle5 as pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


data = pd.read_csv('./data/CompleteDataset.csv', sep=',', on_bad_lines="skip")

arrs = [[], [], [], [], []]

for index, rating in enumerate(data["rating"]):
    arrs[int(rating) - 1].append(str(data["review"][index]))

print(len(arrs[0]), len(arrs[1]), len(arrs[2]), len(arrs[3]), len([arrs[4]]))


max_length = 500
trunc_type='post'
padding_type='post'

model = tf.keras.models.load_model('./my_model')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

avarages = []

for sentences in arrs:
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    prediction = model.predict(padded)
    predictions = prediction.flatten()

    avarages.append(predictions.sum() / len(sentences))

print(avarages)

