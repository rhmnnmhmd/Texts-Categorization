# # Libraries
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout, Masking, Bidirectional, Embedding
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

import matplotlib.pyplot as plt

import os

import datetime as dt

import pandas as pd

import sklearn 
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

import numpy as np

import pickle

from modules import create_model, plot_performance, generate_key_metrics, give_prediction, conv


# # Statics
time_stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
LOG_PATH = os.path.join(os.getcwd(), 'logs', time_stamp)

# # Load data
DATA_PATH = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
df = pd.read_csv(DATA_PATH)

# ## general infos
df.info()

# ## number of text categories
df['category'].value_counts()

# # Data wrangling
# ## handle duplicates
# ### view how much duplicates
df.duplicated().sum()

# ### view the duplicates itself
df[df.duplicated()]

# ### drop/remove duplicates
df = df.drop_duplicates()
df.duplicated().sum()

# ## remove HTML tags (if any)
df['text'] = df['text'].str.replace('<.*?>', ' ')

# ## remove non-aphabetical characters
df['text'] = df['text'].str.replace('[^a-zA-Z]', ' ')

# ## lowercase all reviews
df['text'] = df['text'].str.lower()

# # Creating features
# ## create features dataframe
texts = df['text']

# ## find the number of words in each text
texts.apply(lambda x: len(x.split()))

# ## boxplot of the number of words in each text
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
texts.apply(lambda x: len(x.split())).plot.box(ax=ax, 
                                               vert=False, 
                                               title='Word numbers in each review', 
                                               grid=True)
plt.tight_layout()
plt.show()

# ## find the median/mean/mode/max/min number of words for the whole given texts
median = texts.apply(lambda x: len(x.split())).median()
mean = texts.apply(lambda x: len(x.split())).mean()
mode = texts.apply(lambda x: len(x.split())).mode()
max_words = texts.apply(lambda x: len(x.split())).max()
min_words = texts.apply(lambda x: len(x.split())).min()
print(f'Median: {median}\nMean: {mean}\nMode: {mode.values[0]}\nMax: {max_words}\nMin: {min_words}')
# ### because there are outliers in the above boxplots, we might want to consider using the median number of words as max padding length

# ## tokenizing
# ### create tokenizer object and fit to the given texts
vocab_size = 10000
oov_token = '<OOV>'
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(texts)

# ### save the fitted tokenizer
TOKENIZER_PATH = os.path.join(os.getcwd(), 'model', 'tokenizer.pkl')
with open(TOKENIZER_PATH, 'wb') as file:
    pickle.dump(tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)

# ### the word index
# print(tokenizer.word_index)

# ### tokenize the given texts
texts_tokenized = tokenizer.texts_to_sequences(texts)

# ## padding the tokenized array
# max length of the padded array (usually depends on the median/mean/mode/min/max number of words per texts)
# we choose median because there are outliers 
max_len = int(texts.apply(lambda x: len(x.split())).median())
texts_tokenized_padded = pad_sequences(texts_tokenized, maxlen=max_len, padding='post', truncating='post')

# # Creating target
# ## encode the category using OneHotEncoder
ohe = OneHotEncoder(sparse=False)
category_encoded = ohe.fit_transform(df[['category']])

# ## categories observed during fitting
print(ohe.categories_)

# ## save the encoder
OHE_PATH = os.path.join(os.getcwd(), 'model', 'ohe.pkl')
with open(OHE_PATH, 'wb') as file:
    pickle.dump(ohe, file)

# # Train-test split
X_train, X_test, y_train, y_test = train_test_split(texts_tokenized_padded, category_encoded, test_size=0.2, random_state=42)

# # Deep Learning
# ## create network layers
input_shape = X_train.shape[-1]
output_shape = y_train.shape[-1]
embedding_dim = 64

# ## create model
model = create_model(in_shape=input_shape, 
                     out_shape=output_shape, 
                     n=128, 
                     act='relu', 
                     drop_rate=0.4, 
                     vocab_size=vocab_size, 
                     embedding_dim=embedding_dim)
model.summary()

## plot model networks
plot_model(model, show_shapes=True, show_layer_names=True)

## compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='acc')

# ## callbacks
tensorboard_callback = TensorBoard(log_dir=LOG_PATH)
early_stopping_callback = EarlyStopping(verbose=1)

# # Model training
# ## train model and save History object
model_hist = model.fit(X_train, 
                       y_train, 
                       epochs=160,
                       batch_size=170,
                       validation_data=(X_test, y_test),
                       callbacks=[tensorboard_callback])

# ## plot loss and metrics against epochs
plot_performance(model_hist)

# # Load Tensorboard. run this two command in terminal
# %load_ext tensorboard
# %tensorboard --logdir logs

# # Model evaluation
model.evaluate(X_test, y_test)

# # Save the model
MODEL_PATH = os.path.join(os.getcwd(), 'model', 'model.h5')
model.save(MODEL_PATH)

# # Predictions
# ## generate decoded predictions from the test data
# predict output
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# vectorize conv(x) function
conve = np.vectorize(conv)

# convert to 'negative' or 'positive'
y_pred_decoded = conve(y_pred).astype(object)

# ## decode test data from one-hot encoded form into original categories
y_test_decoded = ohe.inverse_transform(y_test)
y_test_decoded = y_test_decoded[:,0]

# # Model performance key metrics
generate_key_metrics(y_test_decoded, y_pred_decoded)

# # Custom input testing to let the model predict the text sentiment/category
my_review = input('Give your review: ')

give_prediction(my_review, model, tokenizer, texts)