import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from matplotlib import pyplot
from sklearn import preprocessing
os.makedirs(picture_dir, exist_ok = True)

train_in = np.load(npy_newsdata_dir + 'train_in_SP500_news.npy')
train_out = np.load(npy_newsdata_dir + 'train_out_SP500_news.npy')
test_in = np.load(npy_newsdata_dir + 'test_in_SP500_news.npy')
test_out = np.load(npy_newsdata_dir + 'test_out_SP500_news.npy')

def news_LSTM_Model():
    days = 5
    factor_num = 9

    lstm1 = keras.layers.LSTM(100, activation='tanh', recurrent_activation='hard_sigmoid',
                              use_bias=True, kernel_initializer='glorot_uniform',
                              return_sequences=True)
    dropout1 = keras.layers.Dropout(rate=0.2)
    lstm2 = keras.layers.LSTM(60, activation='tanh', recurrent_activation='hard_sigmoid',
                              use_bias=True, kernel_initializer='glorot_uniform',
                              return_sequences=True)
    lstm3 = keras.layers.LSTM(30, activation='tanh', recurrent_activation='hard_sigmoid',
                              use_bias=True, kernel_initializer='glorot_uniform')
    dropout2 = keras.layers.Dropout(rate=0.2)
    flatten1 = keras.layers.Flatten()
    dense1 = keras.layers.Dense(1, activation=None)

    inputs = keras.Input(shape=(days, factor_num), name='main_input')

    x = lstm1(inputs)
    x = dropout1(x)
    x = lstm2(x)
    x = lstm3(x)
    x = dropout2(x)
    x = flatten1(x)
    outputs = dense1(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


model = news_LSTM_Model()

ini_learning_rate = 0.05
n_epochs = 2000
batch_size = 10
model_name = data_dir + 'LSTM_model.h5'
lrdecay = keras.optimizers.schedules.PolynomialDecay(ini_learning_rate, decay_steps = 10000,
        end_learning_rate = 0.0001, power = 3.0, cycle = False)

model.compile(optimizer = keras.optimizers.Adam(learning_rate = lrdecay), loss = 'mean_squared_error')
hist = model.fit(train_in, train_out, epochs = n_epochs, batch_size = batch_size, validation_data = (test_in, test_out))

model.summary()

model_test_out = model.predict(test_in)

MAX = 2872.8701170000004
MIN = 2532.689941
L = len(model_test_out)
for i in range(L):
    test_out[i] = test_out[i] * (MAX - MIN) + MIN
    model_test_out[i] = model_test_out[i] * (MAX - MIN) + MIN

loss_test_out = model_test_out - test_out
print(loss_test_out)

sum = 0
for i in range(L):
    sum = sum + np.abs(loss_test_out[i]) / np.abs(test_out[i])

print(1 - sum / L)

sum1 = 0
sum2 = 0
for i in range(len(test_out)):
    sum1 = sum1 + square(test_out[i])

for i in range(len(loss_test_out)):
    sum2 = sum2 + square(loss_test_out[i])

sum1 = np.sqrt(sum1)
sum2 = np.sqrt(sum2)
print(1 - sum2 / (sum1 * L))
print(sum2)

fig, ax = plt.subplots(1, 1, figsize = (10, 7))
ax.loglog(hist.history['loss'], ls = '-', lw = 2, alpha = 1, color = 'b' , label = 'Training Data Loss')
ax.loglog(hist.history['val_loss'], ls = '--', lw = 2, alpha = 1, color = 'g' , label = 'Test Data Loss')
ax.set_xlabel('Epochs', fontsize = 10)
ax.set_ylabel('Loss', fontsize = 10)
ax.legend(fontsize = 10)
plt.savefig(picture_dir + 'ELG_SP500_news.png')
