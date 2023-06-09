from keras.models import load_model
import tensorflow as tf
import pandas as pd
import emoji

model = load_model('model.h5')
df = pd.read_csv('reviews.csv')
df['reviews'] = df['reviews'].apply(lambda s: emoji.replace_emoji(s, ' '))
text = df['reviews'].tolist()
token = tf.keras.preprocessing.text.Tokenizer()
token.fit_on_texts(text)

def get_encode(x):
    x = token.texts_to_sequences(x)
    x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=100, padding='post')
    return x

def predict_semantic(text):
    x = get_encode(text)
    pred = model.predict(x)
    return 'negatif' if pred[0][0] > pred[0][1] else 'positif'