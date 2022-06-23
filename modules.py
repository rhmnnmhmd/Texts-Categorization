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


def conv(x):
    """Function to decode numerical predicted output (i.e. 0, 1, 2, 3, 4) into the appropriate categories

    Args:
        x (integer): the numerical value of predicted category

    Returns:
        string: decoded predicted category
    """
    my_dict = {0: 'business',
               1: 'entertainment',
               2: 'politics',
               3: 'sport',
               4: 'tech'}
  
    return my_dict[x]


def create_model(in_shape=333, out_shape=5, n=128, act='relu', drop_rate=0.4, vocab_size=10000, embedding_dim=64):
    """Function to create the neural network model

    Args:
        in_shape (int, optional): the input shape of the model. Defaults to 333.
        out_shape (int, optional): the output shape of the model. Defaults to 5.
        n (int, optional): number of units/cells in a layer. Defaults to 128.
        act (str, optional): the activation function for the layer. Defaults to 'relu'.
        drop_rate (float, optional): dropout rate. Defaults to 0.4.
        vocab_size (int, optional): the estimated number of unique words in the input text corpus. Defaults to 10000.
        embedding_dim (int, optional): embedding size for the embedding layer. Defaults to 64.

    Returns:
        Tensorflow Model object: the neural network model
    """
    input_1 = Input(shape=(in_shape))
    hl_0 = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_1)
    hl_1 = Bidirectional(LSTM(embedding_dim, return_sequences=True))(hl_0)
    hl_2 = Dropout(drop_rate)(hl_1)
    hl_3 = LSTM(n)(hl_2)
    hl_4 = Dropout(drop_rate)(hl_3)
    hl_5 = Dense(n, activation=act)(hl_4)
    hl_6 = Dropout(drop_rate)(hl_5)       
    output_1 = Dense(output_shape, activation='softmax')(hl_6)

    return Model(inputs=input_1, outputs=output_1)


def plot_performance(model_hist):
    """Plot the loss and metric curve against the epochs for both training and test set

    Args:
        model_hist (Tensorflow History object): Tensorflow History object obtained from using .fit() method to a Tensorflow Model object
    """
    train_loss = model_hist.history['loss']
    train_metric = model_hist.history['acc']
    test_loss = model_hist.history['val_loss']
    test_metric = model_hist.history['val_acc']

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Training Loss and Accuracy vs Epochs')

    ax[0].plot(train_loss, label='training loss')
    ax[0].plot(test_loss, label='test loss')
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('categorical cross-entropy loss')
    ax[0].legend()

    ax[1].plot(train_metric, label='train accuracy')
    ax[1].plot(test_metric, label='test accuracy')
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('accuracy')
    ax[1].legend()

    plt.tight_layout()
    plt.show()


def generate_key_metrics(y_true, y_pred):
    """Generate key metrics of the model e.g. classification report and confusion matrix

    Args:
        y_true (numpy array): 1-D array of the true target label
        y_pred (numpy array): 1-D array of the predicted target label
    """
    # # Model performance key metrics
    # ## classification report
    report = classification_report(y_true, y_pred)
    print(report)

    # ## confusion matrix plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize='all', ax=ax)
    plt.tight_layout()
    plt.show()


def give_prediction(my_review, model, tokenizer, texts):
    """Predict the sentiment/category of the texts inputted by the user

    Args:
        my_review (string): The input texts
        model (Tensorflow Model object): Tensorflow fitted Model object
        tokenizer (_type_): Fitted tokenizer
        texts (_type_): the original training texts
    """
    # preprocess the inputted texts
    input_review = pd.Series([my_review], name='review')
    input_review = input_review.str.replace('<.*?>', ' ')
    input_review = input_review.str.replace('[^a-zA-Z]', ' ')
    input_review = input_review.str.lower()

    # tokenize the input review
    input_review_tokenized = tokenizer.texts_to_sequences(input_review)

    # padding the tokenized review
    max_len = int(texts.apply(lambda x: len(x.split())).median()) # this value is 333 for now
    input_review_tokenized_padded = pad_sequences(input_review_tokenized, maxlen=max_len, padding='post', truncating='post')

    # generate the predicted category
    predicted_sentiment = model.predict(input_review_tokenized_padded)
    predicted_sentiment = np.argmax(predicted_sentiment, axis=1)
    conve = np.vectorize(conv)
    predicted_sentiment_decoded = conve(predicted_sentiment).astype(object)

    # print the predicted category
    print(f'Your review have {predicted_sentiment_decoded[0]} sentiment')