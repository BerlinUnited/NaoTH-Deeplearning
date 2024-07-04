"""
    Refer to https://www.tensorflow.org/lite/performance/post_training_quantization
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


def get_model(model):
    # Load your Keras model
    return load_model(model)


def representative_dataset_from_h5(dataset_path, rescale=False, subtract_mean=False):
    with h5py.File(dataset_path, "r") as f:
        X = f["X"][:]

    X = X.astype(np.float32)

    if rescale:
        X = X / 255.0

    if subtract_mean:
        X = X - np.mean(X, axis=0)

    return representative_dataset_from_np_arr(X)


def representative_dataset_from_np_arr(arr, size=2048):
    arr_copy = arr.copy()
    np.random.shuffle(arr_copy)

    def representative_dataset_gen():
        for x in tf.data.Dataset.from_tensor_slices((arr_copy)).batch(1).take(size):
            yield [tf.dtypes.cast(x, tf.float32)]

    return representative_dataset_gen


def validate_model_file_format(model_path):
    assert str(model_path).endswith(".h5") or str(model_path).endswith(
        ".keras"
    ), "Model file must be in .h5 or .keras format!"


def convert_to_float32_tflite_model(model_path):
    """
    Convert to float32 TFLite model, no quantization.

    https://www.tensorflow.org/lite/models/convert/convert_models#convert_a_keras_model_
    """
    validate_model_file_format(model_path)
    model = get_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()

    output_file = Path(model_path).with_suffix(".tflite")
    with open(output_file, "wb") as f:
        f.write(tflite_model)


def convert_to_int_weights_mixed_ops_float_fallback_tflite_model(model_path):
    """
    Parameters are quantized to integers. Model executes with operations that mix
    integer and float computation when available; otherwise falls back to float32
    operation.

    https://www.tensorflow.org/lite/performance/post_training_quantization#dynamic_range_quantization
    """

    validate_model_file_format(model_path)
    model = get_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    output_file = Path(model_path).parent / (
        str(Path(model_path).stem) + "_int_weights_mixed_ops_float_fallback.tflite"
    )
    with open(output_file, "wb") as f:
        f.write(tflite_model)


def convert_to_int_weights_int_ops_float_fallback_tflite_model(model_path, representative_dataset):
    """
    Integer quantization with float fallback (using default float input/output).

    Parameters and activations are quantized to integers. Model executes
    with integer operations when available; otherwise falls back to float32
    operation.

    https://www.tensorflow.org/lite/performance/post_training_quantization#integer_with_float_fallback_using_default_float_inputoutput

    """
    validate_model_file_format(model_path)
    model = get_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()

    output_file = Path(model_path).parent / (str(Path(model_path).stem) + "_int_weights_int_ops_float_fallback.tflite")

    with open(output_file, "wb") as f:
        f.write(tflite_model)


def convert_to_fully_int_quantized_tflite_model(model_path, representative_dataset):
    """
    Full integer quantization for all ops, including the input and output.

    Parameters and activations are quantized to integers. Model executes with integer
    operations only. Conversion fails if the model has unsupported operation.

    https://www.tensorflow.org/lite/performance/post_training_quantization#integer_only
    """
    validate_model_file_format(model_path)
    model = get_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()

    output_file = Path(model_path).parent / (str(Path(model_path).stem) + "_full_int_quant.tflite")
    with open(output_file, "wb") as f:
        f.write(tflite_model)


def convert_to_float16_quantized_tflite_model(model_path, representative_dataset):
    """
    Parameters are quantized to float16. Model executes with float32 operations.

    NOTE: By default, a float16 quantized model will "dequantize" the weights values to
    float32 when run on the CPU. (A GPU delegate will not perform this dequantization,
    since it can operate on float16 data.)

    https://www.tensorflow.org/lite/performance/post_training_quantization#float16_quantization
    """
    validate_model_file_format(model_path)
    model = get_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()

    output_file = Path(model_path).parent / (str(Path(model_path).stem) + "_float16_quant.tflite")
    with open(output_file, "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-ds", "--dataset", required=True)
    args = parser.parse_args()

    # fmt: off
    convert_to_float32_tflite_model(args.model)
    convert_to_int_weights_mixed_ops_float_fallback_tflite_model(args.model)
    convert_to_int_weights_int_ops_float_fallback_tflite_model(args.model, representative_dataset_from_h5(args.dataset))
    convert_to_fully_int_quantized_tflite_model(args.model, representative_dataset_from_h5(args.dataset))
    convert_to_float16_quantized_tflite_model(args.model, representative_dataset_from_h5(args.dataset))
    # fmt: on
