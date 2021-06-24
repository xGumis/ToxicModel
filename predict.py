def pred(str):
	import os
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	import numpy as np
	import tensorflow as tf
	import json
	from tensorflow import keras
	from keras.preprocessing.sequence import pad_sequences
	from keras.preprocessing.text import tokenizer_from_json

	MAX_LEN = 1400
	MAX_FEATURES = 5000

	text = [str]

	with open('token.json') as f:
		data = json.load(f)

	tokenizer = tokenizer_from_json(data)
	text = tokenizer.texts_to_sequences(text)
	text = pad_sequences(text, maxlen=MAX_LEN)

	model = keras.models.load_model('gru_model')

	# toxic  severe_toxic  obscene  threat  insult  identity_hate
	return model.predict(text)