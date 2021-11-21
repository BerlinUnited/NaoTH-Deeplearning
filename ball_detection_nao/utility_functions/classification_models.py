"""
    NaoTH Classification Models
"""
import inspect

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, ReLU, MaxPooling2D, Flatten, Dense

def naoth_classification1():
    """
    """
    input_shape = (16, 16, 1)
    model = Sequential()
    model._name = inspect.currentframe().f_code.co_name  # get the name of the function

    # we don't know the kernel size b-human used
    model.add(Convolution2D(16, (3, 3), input_shape=input_shape, padding='same', name="Conv2D_2"))
    # Batch Norm
    model.add(ReLU(name="activation_2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_2"))

    # we don't know the kernel size b-human used
    model.add(Convolution2D(16, (3, 3), padding='same', name="Conv2D_3"))
    # Batch Norm
    model.add(ReLU(name="activation_3"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_3"))

    # we don't know the kernel size b-human used
    model.add(Convolution2D(32, (3, 3), padding='same', name="Conv2D_4"))
    # Batch Norm
    model.add(ReLU(name="activation_4"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_4"))
    model.add(Flatten(name="flatten_1"))
    model.add(Dense(32, activation="relu", name="dense_1"))
    model.add(Dense(64, activation="relu", name="dense_2"))
    model.add(Dense(16, activation="relu", name="dense_3"))
    model.add(Dense(1, activation="relu", name="dense_4"))

    # For using custom loss import your loss function and use the name of the function as loss argument.
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model


def naoth_classification2():
    """
        use bigger patches here. This is needed to train on original b-human data
    """
    input_shape = (32, 32, 1)
    model = Sequential()
    model._name = inspect.currentframe().f_code.co_name  # get the name of the function

    # we don't know the kernel size b-human used
    model.add(Convolution2D(16, (3, 3), input_shape=input_shape, padding='same', name="Conv2D_2"))
    # Batch Norm
    model.add(ReLU(name="activation_2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_2"))

    # we don't know the kernel size b-human used
    model.add(Convolution2D(16, (3, 3), padding='same', name="Conv2D_3"))
    # Batch Norm
    model.add(ReLU(name="activation_3"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_3"))

    # we don't know the kernel size b-human used
    model.add(Convolution2D(32, (3, 3), padding='same', name="Conv2D_4"))
    # Batch Norm
    model.add(ReLU(name="activation_4"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_4"))
    model.add(Flatten(name="flatten_1"))
    model.add(Dense(32, activation="relu", name="dense_1"))
    model.add(Dense(64, activation="relu", name="dense_2"))
    model.add(Dense(16, activation="relu", name="dense_3"))
    model.add(Dense(1, activation="relu", name="dense_4"))

    # For using custom loss import your loss function and use the name of the function as loss argument.
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model