from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, LeakyReLU, MaxPooling2D, Flatten, Dense, ReLU, Input, \
    Softmax, concatenate, Dropout, UpSampling2D, BatchNormalization
from tensorflow.keras.activations import sigmoid
from tensorflow.keras import Model
import inspect


def fy_1500_new():
    """
    The idea here is to remove the relu in the last layer. That makes sure that the x and y values can be negative
    :return: model
    """
    input_shape = (16, 16, 1)

    model = Sequential()
    model._name = inspect.currentframe().f_code.co_name  # get the name of the function

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

    # For using custom loss import your loss function and use the name of the function as loss argument.
    # FIXME: devil compiler 1 kann mit custom metrics nicht umgehen
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model


def fy_1500_new2():
    """
    The idea here is to remove the relu in the last layer. That makes sure that the x and y values can be negative
    :return: model
    """
    input_shape = (16, 16, 1)

    model = Sequential()
    model._name = inspect.currentframe().f_code.co_name  # get the name of the function

    model.add(Convolution2D(4, (3, 3), input_shape=input_shape, padding='same', name="Conv2D_1"))
    model.add(LeakyReLU(alpha=0.0, name="activation_1"))  # alpha unknown, so default
    model.add(Convolution2D(4, (3, 3), padding='same', name="Conv2D_2"))
    model.add(LeakyReLU(alpha=0.0, name="activation_2"))  # alpha unknown, so default
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(8, (3, 3), padding='same', name="Conv2D_3"))
    model.add(LeakyReLU(alpha=0.0, name="activation_3"))  # alpha unknown, so default
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_1"))
    model.add(Convolution2D(8, (3, 3), padding='same', name="Conv2D_4"))
    model.add(LeakyReLU(alpha=0.0, name="activation_4"))  # alpha unknown, so default
    model.add(Convolution2D(8, (1, 1), padding='same', name="Conv2D_5"))

    # classifier
    model.add(Flatten(name="flatten_1"))
    # output is radius, x, y, confidence
    model.add(Dense(4, name="dense_1"))

    # For using custom loss import your loss function and use the name of the function as loss argument.
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model