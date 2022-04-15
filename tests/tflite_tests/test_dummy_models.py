"""

"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
import time
import tensorflow.keras as keras


def test_dummy2_model_tflite():
    # test output
    input_data = np.ones((1, 16, 16, 1), dtype=np.float32)

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="dummy_model2.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data

    t0 = time.time()
    for i in range(1000):
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
    t1 = time.time()

    total = (t1 - t0) * 1000
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(total, "milliseconds")


def test_dummy2_model_keras():
    # test output
    input_data = np.ones((1, 16, 16, 1), dtype=np.float32)

    model = keras.models.load_model("dummy_model2.h5")

    t0 = time.time()
    for i in range(1000):
        out = model.predict(input_data)
    t1 = time.time()

    total = (t1 - t0) * 1000

    print(total, "milliseconds for keras")


if __name__ == '__main__':
    #test_dummy2_model_tflite()
    test_dummy2_model_keras()
