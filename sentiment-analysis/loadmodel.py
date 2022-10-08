import numpy as np
import pandas as pd
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

max_length = 500
trunc_type='post'
padding_type='post'

model = tf.keras.models.load_model('./my_model')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


sentence_arr = ["Very good. Best thing I have ever seen. Period.", "Worst crap garbage. Shit fuck it was bad!"]

sequences = tokenizer.texts_to_sequences(sentence_arr)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
prediction = model.predict(padded)
predictions = prediction.flatten()

predictions_mapped = []
for p in predictions:
    if p > 0.04:
        predictions_mapped.append(4)
    else:
        predictions_mapped.append(2)

print(predictions_mapped)

