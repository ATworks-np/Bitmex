from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda, Reshape
from keras.layers.convolutional import Conv2D, Deconv2D, ZeroPadding2D, UpSampling2D
from keras.layers import Input, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
import keras.backend as K
import numpy as np
from keras.layers.recurrent import LSTM

def my_lstm(input_shape, action_shape):
    n_hidden = 50
    input = Input(shape=input_shape, name="XBT_input")
    input2 = Input(shape=[action_shape], name="position_input")
    x = LSTM(n_hidden, batch_input_shape=(None, input_shape[0], input_shape[1]), return_sequences=False)(input)
    x = Concatenate()([x,input2])
    x = Dense(100, activation='relu')(x)
    x = Dense(action_shape, activation='softmax')(x)

    model = Model(inputs=[input,input2], outputs=[x], name='my_lstm')
    model.summary()

    return model

def load_lstm(input_shape, action_shape):
    model = my_lstm(input_shape, action_shape)
    model.summary()
    return model

load_lstm([12,4],3)
