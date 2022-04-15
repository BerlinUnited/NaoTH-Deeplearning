"""
    Code for converting our trained keras models (.h5) to tflite models. The output file will be saved
    to same folder with the .tflite suffix.
"""
import tensorflow as tf
import tensorflow.keras as keras
from pathlib import Path


def convert_model(input_file_path):
    model = keras.models.load_model(input_file_path)

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model
    output_path = Path(input_file_path).with_suffix(".tflite")
    with open(str(output_path), 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    convert_model("fy1500.h5")
