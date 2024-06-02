"""
    Refer to https://www.tensorflow.org/lite/performance/post_training_quantization
"""

from tensorflow.keras.models import load_model
import tensorflow as tf
import argparse
import h5py
from pathlib import Path


def get_model(model):
    # Load your Keras model
    return load_model(model)


def representative_dataset(dataset_path):
    def representative_dataset_gen():
        with h5py.File(dataset_path, "r") as f:
            for idx, a in enumerate(f['X']):
                image = f['X'][idx]
                yield {
                "image": image
                }
    return representative_dataset_gen

def generate_fully_dynamic_quantized_model(model_path, dataset_path):
    # TODO: check that it has h5 or .keras file ending
    model = get_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(dataset_path)

    tflite_model = converter.convert()
    
    output_file = Path(model_path).with_suffix(".tflite")
    with open(output_file, 'wb') as f:
        f.write(tflite_model)


def generate_fully_int_quantized_model(model_path, dataset_path):
    model = get_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    converter.representative_dataset = representative_dataset(dataset_path)

    tflite_model = converter.convert()

    output_file = Path(model_path).parent / (str(Path(model_path).stem) + "_int_quant.tflite")
    with open(output_file, 'wb') as f:
        f.write(tflite_model)


def generate_float16_quantized_model(model_path, dataset_path):
    model = get_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.representative_dataset = representative_dataset(dataset_path)

    tflite_model = converter.convert()

    output_file = Path(model_path).parent / (str(Path(model_path).stem) + "_float16_quant.tflite")
    with open(output_file, 'wb') as f:
        f.write(tflite_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-ds", "--dataset", required=True)
    args = parser.parse_args()

    generate_fully_dynamic_quantized_model(args.model, args.dataset)
    generate_fully_int_quantized_model(args.model, args.dataset)
    generate_float16_quantized_model(args.model, args.dataset)