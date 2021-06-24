import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')

EPOCHS = 30
MAX_LEN = 1400
MAX_FEATURES = 5000
EMBEDDING_VECTOR_LENGTH = 50
BATCH = 32

dataset = pd.read_csv("jigsaw-toxic-comment-train.csv")

x_train = dataset["comment_text"]
y_train = dataset.iloc[:,2:]

tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(x_train)
token_json = tokenizer.to_json()
with open('token.json', 'w',encoding="utf-8") as f:
    json.dump(token_json,f, ensure_ascii=False)
	
# x_train = tokenizer.texts_to_sequences(x_train)
# x_train = pad_sequences(x_train, maxlen=MAX_LEN)

# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

# ft = fasttext.load_model('cc.en.300.bin')
# fasttext.util.reduce_model(ft, 50)
# embedding_matrix_fasttext = np.zeros((MAX_FEATURES + 1, EMBEDDING_VECTOR_LENGTH))

# for word, i in sorted(tokenizer.word_index.items(),key=lambda x:x[1]):
    # if i > (MAX_FEATURES+1):
        # break
    # try:
        # embedding_vector = ft[word] #Reading word's embedding from Glove model for a given word
        # embedding_matrix_fasttext[i] = embedding_vector
    # except:
        # pass
		
# model = keras.Sequential()
# model.add(keras.layers.Embedding(MAX_FEATURES+1,
                    # EMBEDDING_VECTOR_LENGTH, ### 50 here
                    # weights=[embedding_matrix_fasttext],
                    # input_length=MAX_LEN, ### 1400 here
                    # trainable=False))
# model.add(keras.layers.SpatialDropout1D(0.3))
# model.add(keras.layers.GRU(300))
# model.add(keras.layers.Dense(6, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy',tf.keras.metrics.AUC()])
# model.summary()

# history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH, validation_data=(x_val, y_val))

# with open('data.json', 'w') as f:
    # json.dump(history.history, f)

# model.save("gru_model")