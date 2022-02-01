"""
    creates keras h5 model that is later used for testing frugally exporter
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense


def fy_1500_new():
    # TODO maybe i can import the function rather than copying it
    """
    This function is taken from model_zoo for nao ball detection

    The idea here is to remove the relu in the last layer. That makes sure that the x and y values can be negative
    :return: model
    """
    input_shape = (16, 16, 1)

    model = Sequential()
    model._name = "test_model"

    model.add(Convolution2D(4, (3, 3), input_shape=input_shape, padding='same', name="Conv2D_1", activation='relu'))
    model.add(Convolution2D(4, (3, 3), padding='same', name="Conv2D_2", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(8, (3, 3), padding='same', name="Conv2D_3", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_1"))
    model.add(Convolution2D(8, (3, 3), padding='same', name="Conv2D_4", activation='relu'))
    model.add(Convolution2D(8, (1, 1), padding='same', name="Conv2D_5"))

    # classifier
    model.add(Flatten(name="flatten_1"))
    # output is radius, x, y, confidence
    model.add(Dense(4, name="dense_1"))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

model = fy_1500_new()

model.save('keras__test_model.h5', include_optimizer=False)