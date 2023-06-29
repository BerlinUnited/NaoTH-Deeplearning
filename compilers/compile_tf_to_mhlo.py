"""
    Needs tensorflow 2.8.0
    Taken from emitc repo
    1. convert saved model to tf dialect
    2. convert tf dialect to mhlo
    3. run emitc
    4. profit

"""
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python import pywrap_mlir
from pathlib import Path
import numpy as np
import random


def set_fake_weights(model):
    for layer in model.layers:
        if layer.get_weights():
            new_weights = []
            for weight in layer.get_weights():
                const_weight = np.full(weight.shape, 0.5)
                new_weights.append(const_weight)
            layer.set_weights(new_weights)
    return model


class Module(tf.Module):
    def __init__(self, model):
        super().__init__()
        self._model = model

    def predict(self, *args):
        return self._model.call(list(args), training=False)


def extract_tensor_specs(model, batch_size: int):
    def extract_tensor_spec(tensor):
        shape = list(tensor.shape)

        if shape[0] is None:
            shape[0] = batch_size

        return tf.TensorSpec(shape, tensor.dtype)

    return [extract_tensor_spec(tensor) for tensor in model.inputs]


def create_keras_model():
    model = MobileNetV2(weights='imagenet')
    set_fake_weights(model)

    model.save("mobilenet_v2.h5")


def convert_to_saved_model(input_file_path):
    model = keras.models.load_model(input_file_path)
    # Produce a concrete function to compile.
    module = Module(model)
    module.predict = tf.function(
        func=module.predict,
        input_signature=extract_tensor_specs(model, batch_size=1),
    )

    tf.saved_model.save(module, "mobilenet_v2")


def convert_to_tf_dialect():
    exported_names = "predict"
    with open("dummy_model_tf_dialect.mlir", "w") as file:
        file.write(
            pywrap_mlir.experimental_convert_saved_model_to_mlir(
                "mobilenet_v2", exported_names, True))


def convert_to_mhlo():
    with open("dummy_model_tf_dialect.mlir") as file:
        mlir = file.read()

    pass_pipeline = ",".join([
        "symbol-dce", "tf-standard-pipeline",
        "builtin.func(tf-device-index-selector)", "inline", "canonicalize",
        "builtin.func(tf-device-decompose-resource-ops)",
        "builtin.func(tf-functional-control-flow-to-cfg)", "inline", "symbol-dce",
        "canonicalize", "tf-saved-model-optimize-global-tensors",
        "tf-saved-model-freeze-global-tensors"
    ])
    #with open("dummy_model_mhlo.mlir", "w") as file:
    #    file.write(
    #        pywrap_mlir.experimental_run_pass_pipeline(mlir, pass_pipeline, True))
    mlir = pywrap_mlir.experimental_run_pass_pipeline(mlir, pass_pipeline, True)

    pass_pipeline = ",".join([
        "builtin.func(xla-legalize-tf)", "canonicalize",
        "tf-saved-model-optimize-global-tensors"
    ])

    with open("dummy_model_mhlo.mlir", "w") as file:
        file.write(
            pywrap_mlir.experimental_run_pass_pipeline(mlir, pass_pipeline, True))


if __name__ == '__main__':
    #create_keras_model()
    convert_to_saved_model("mobilenet_v2.h5")
    convert_to_tf_dialect()
    convert_to_mhlo()
