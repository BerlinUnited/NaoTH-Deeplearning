import shutil
from pathlib import Path

import tensorflow as tf

model_path = "data/BOTTOM/trionda_small.keras"

model = tf.keras.models.load_model(model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

output_file = Path(model_path).with_suffix(".tflite")
with open(output_file, "wb") as f:
    f.write(tflite_model)
