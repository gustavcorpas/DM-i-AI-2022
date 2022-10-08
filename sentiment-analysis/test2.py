
import numpy as np
import pandas as pd
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


vocab_size = 20000
embedding_dim = 32
max_length = 500
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 40000
num_epochs = 6

data = pd.read_csv('./data/CompleteDataset.csv', sep=',', on_bad_lines="skip")

sentences = []
labels = []

for item in data["review"]:
    if len(str(item)) < 4:
        print("error on: ", item)
    sentences.append(str(item))

for item in data["rating"]:
    
    if item == 1 or item == 2:
        labels.append(0)
    elif item == 3 or item == 4 or item == 5:
        labels.append(1)
    else:
        print("ERROR on: ", item)

print(len(sentences), len(labels))

training_sentences = sentences[:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[:training_size]
testing_labels = labels[training_size:]


tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=1)


sentence_arr = pd.read_csv('./data/data.csv', sep=',', on_bad_lines="skip")

sequences = tokenizer.texts_to_sequences(sentence_arr["comment"])
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(model.predict(padded))
model.save('./my_model')

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
