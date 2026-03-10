import tensorflow as tf
from pathlib import Path
import shutil 

model_path = "data/BOTTOM/trionda_small.keras"
temp_export_dir = "data/BOTTOM/temp_saved_model"

model = tf.keras.models.load_model(model_path)

model.export(temp_export_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(temp_export_dir)
tflite_model = converter.convert()

output_file = Path(model_path).with_suffix(".tflite")
with open(output_file, "wb") as f:
    f.write(tflite_model)

shutil.rmtree(temp_export_dir, ignore_errors=True)