"""
    NaoTH Classification Models
"""
import inspect

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Convolution2D, ReLU, MaxPooling2D, Flatten, Dense, Concatenate, BatchNormalization, Softmax

def naoth_classification_8_8():
    """
        use bigger patches here. This is needed to train on original b-human data
    """
    input_shape = (8, 8, 1)
    model = Sequential()
    model._name = inspect.currentframe().f_code.co_name  # get the name of the function

    # we don't know the kernel size b-human used
    model.add(Convolution2D(16, (3, 3), input_shape=input_shape, padding='same', name="Conv2D_1"))
    # Batch Norm
    model.add(ReLU(name="activation_1"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_1"))

    # we don't know the kernel size b-human used
    model.add(Convolution2D(16, (3, 3), padding='same', name="Conv2D_2"))
    # Batch Norm
    model.add(ReLU(name="activation_2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_2"))

    # we don't know the kernel size b-human used
    model.add(Convolution2D(32, (3, 3), padding='same', name="Conv2D_3"))
    # Batch Norm
    model.add(ReLU(name="activation_3"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_3"))
    model.add(Flatten(name="flatten_1"))
    model.add(Dense(32, activation="relu", name="dense_1"))
    model.add(Dense(64, activation="relu", name="dense_2"))
    model.add(Dense(16, activation="relu", name="dense_3"))
    model.add(Dense(1, activation="relu", name="dense_4"))

    # For using custom loss import your loss function and use the name of the function as loss argument.
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model


def naoth_classification_8_8_color():
    """
        use bigger patches here. This is needed to train on original b-human data
    """
    input_shape = (8, 8, 3)
    model = Sequential()
    model._name = inspect.currentframe().f_code.co_name  # get the name of the function

    # we don't know the kernel size b-human used
    model.add(Convolution2D(16, (3, 3), input_shape=input_shape, padding='same', name="Conv2D_1"))
    # Batch Norm
    model.add(ReLU(name="activation_1"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_1"))

    # we don't know the kernel size b-human used
    model.add(Convolution2D(16, (3, 3), padding='same', name="Conv2D_2"))
    # Batch Norm
    model.add(ReLU(name="activation_2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_2"))

    # we don't know the kernel size b-human used
    model.add(Convolution2D(32, (3, 3), padding='same', name="Conv2D_3"))
    # Batch Norm
    model.add(ReLU(name="activation_3"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_3"))
    model.add(Flatten(name="flatten_1"))
    model.add(Dense(32, activation="relu", name="dense_1"))
    model.add(Dense(64, activation="relu", name="dense_2"))
    model.add(Dense(16, activation="relu", name="dense_3"))
    model.add(Dense(1, activation="relu", name="dense_4"))

    # For using custom loss import your loss function and use the name of the function as loss argument.
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model

def naoth_classification_16_16():
    """
    """
    input_shape = (16, 16, 1)
    model = Sequential()
    model._name = inspect.currentframe().f_code.co_name  # get the name of the function

    # we don't know the kernel size b-human used
    model.add(Convolution2D(16, (3, 3), input_shape=input_shape, padding='same', name="Conv2D_1"))
    # Batch Norm
    model.add(ReLU(name="activation_1"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_1"))

    # we don't know the kernel size b-human used
    model.add(Convolution2D(16, (3, 3), padding='same', name="Conv2D_2"))
    # Batch Norm
    model.add(ReLU(name="activation_2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_2"))

    # we don't know the kernel size b-human used
    model.add(Convolution2D(32, (3, 3), padding='same', name="Conv2D_3"))
    # Batch Norm
    model.add(ReLU(name="activation_3"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_3"))
    model.add(Flatten(name="flatten_1"))
    model.add(Dense(32, activation="relu", name="dense_1"))
    model.add(Dense(64, activation="relu", name="dense_2"))
    model.add(Dense(16, activation="relu", name="dense_3"))
    model.add(Dense(1, activation="relu", name="dense_4"))

    # For using custom loss import your loss function and use the name of the function as loss argument.
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model


def naoth_classification_32_32():
    """
        use bigger patches here. This is needed to train on original b-human data
    """
    input_shape = (32, 32, 1)
    model = Sequential()
    model._name = inspect.currentframe().f_code.co_name  # get the name of the function

    # we don't know the kernel size b-human used
    model.add(Convolution2D(16, (3, 3), input_shape=input_shape, padding='same', name="Conv2D_1"))
    # Batch Norm
    model.add(ReLU(name="activation_1"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_1"))

    # we don't know the kernel size b-human used
    model.add(Convolution2D(16, (3, 3), padding='same', name="Conv2D_2"))
    # Batch Norm
    model.add(ReLU(name="activation_2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_2"))

    # we don't know the kernel size b-human used
    model.add(Convolution2D(32, (3, 3), padding='same', name="Conv2D_3"))
    # Batch Norm
    model.add(ReLU(name="activation_3"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_3"))
    model.add(Flatten(name="flatten_1"))
    model.add(Dense(32, activation="relu", name="dense_1"))
    model.add(Dense(64, activation="relu", name="dense_2"))
    model.add(Dense(16, activation="relu", name="dense_3"))
    model.add(Dense(1, activation="relu", name="dense_4"))

    # For using custom loss import your loss function and use the name of the function as loss argument.
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model

def naoth_classification_64_64():
    """
        use bigger patches here. This is needed to train on original b-human data
    """
    input_shape = (64, 64, 1)
    model = Sequential()
    model._name = inspect.currentframe().f_code.co_name  # get the name of the function

    # we don't know the kernel size b-human used
    model.add(Convolution2D(16, (3, 3), input_shape=input_shape, padding='same', name="Conv2D_1"))
    # Batch Norm
    model.add(ReLU(name="activation_1"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_1"))

    # we don't know the kernel size b-human used
    model.add(Convolution2D(16, (3, 3), padding='same', name="Conv2D_2"))
    # Batch Norm
    model.add(ReLU(name="activation_2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_2"))

    # we don't know the kernel size b-human used
    model.add(Convolution2D(32, (3, 3), padding='same', name="Conv2D_3"))
    # Batch Norm
    model.add(ReLU(name="activation_3"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="pooling_3"))
    model.add(Flatten(name="flatten_1"))
    model.add(Dense(32, activation="relu", name="dense_1"))
    model.add(Dense(64, activation="relu", name="dense_2"))
    model.add(Dense(16, activation="relu", name="dense_3"))
    model.add(Dense(1, activation="relu", name="dense_4"))

    # For using custom loss import your loss function and use the name of the function as loss argument.
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model

def hulks_ball_classification_model():
    """
        taken from the hulks team report from 2019 https://hulks.de/_files/TRR_2019.pdf
    """
    input_shape = (15, 15, 3)

    inputs = keras.Input(shape=input_shape)
    # TODO add documentation on how I found this architecture
    # i put the network found in https://github.com/HULKs/HULKsCodeRelease/tree/main/home/neuralnets into netron.app to 
    # see the architecture. Another overview can be found in the team report at https://hulks.de/_files/TRR_2019.pdf
    x1 = Convolution2D(2, (7, 7), padding='same', name="Conv2D_1")(inputs)
    x2 = Convolution2D(2, (5, 5), padding='same', name="Conv2D_2")(inputs)
    x3 = Convolution2D(2, (3, 3), padding='same', name="Conv2D_3")(inputs)
    concatted = Concatenate(axis=3)([x1, x2, x3])
    x = BatchNormalization()(concatted)
    x = Convolution2D(5, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Convolution2D(5, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Convolution2D(5, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Convolution2D(5, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Convolution2D(5, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Convolution2D(5, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Convolution2D(2, (3, 3))(x)
    outputs = Softmax()(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=inspect.currentframe().f_code.co_name)
    model.summary()
