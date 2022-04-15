"""
    creates tflite models which can be used to check correctness after compiling it with various compilers
    NOTE: each run currently produces new weights and therefor different models.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense


def create_dummy1_model():
    """
    simple model which just adds the input to the float number 42. This model expects that everything is in float32.
    This is important when using the tflite libs. If int8 quantization is enabled additional pre and post prcessing
    on the c++ part is needed.
    """
    a = tf.keras.layers.Input(shape=(1,), dtype=np.float32)
    b = a + tf.constant(42.)

    model = keras.Model(inputs=a, outputs=b, name="dummy_model1")
    model.compile()

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    # converter.representative_dataset = representative_dataset
    # converter.inference_input_type = tf.int8
    # converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    # Save the model.
    with open('dummy_model1.tflite', 'wb') as f:
        f.write(tflite_model)


def create_dummy2_model():
    """
    copied from fy_1500_new from model_zoo in the ball_detection folder. This is only for creating a tflite model to
    do timing tests with tflite and tflite-micro
    """
    input_shape = (16, 16, 1)

    model = Sequential()
    model._name = "dummy_ball_detection_model"

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

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the model.
    with open('dummy_model2.tflite', 'wb') as f:
        f.write(tflite_model)

    model.save("dummy_model2.h5", save_format='h5')


def create_dummy3_model():
    """
    simple model which just adds the input to the float number 42. But it used int8 quantization in the process

    TODO: this is not tested yet. Maybe additional steps are necessary to make it work in c++
    """
    a = tf.keras.layers.Input(shape=(1,), dtype=np.float32)
    b = a + tf.constant(42.)

    model = keras.Model(inputs=a, outputs=b, name="dummy_model1")
    model.compile()

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # representative dataset is needed in order to figure out the min and max values
    converter.representative_dataset = representative_dataset3
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    # Save the model.
    with open('dummy_model3.tflite', 'wb') as f:
        f.write(tflite_model)


def representative_dataset3():
    for _ in range(100):
        data = np.random.rand(1, )
        yield [data.astype(np.float32)]


if __name__ == '__main__':
    create_dummy1_model()
    create_dummy2_model()
    create_dummy3_model()
