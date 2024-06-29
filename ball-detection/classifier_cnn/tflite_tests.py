import tensorflow as tf
from pathlib import Path
from losses import WeightedBinaryCrossentropy
from utils import load_h5_dataset_X_y
import numpy as np

class TFLiteModel:
    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, *data_args):
        assert len(data_args) == len(self.input_details)
        for data, details in zip(data_args, self.input_details):
            self.interpreter.set_tensor(details["index"], data)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]["index"])

data_root="../../data"
data_train = Path(data_root) / "classification_patches_yuv422_y_only_pil_legacy_border0/classification_patches_yuv422_y_only_pil_legacy_border0_combined_16x16_train_ball_no_ball_X_y.h5"
X, Y = load_h5_dataset_X_y(data_train)
X_mean = X.mean()
X = X.astype(np.float32) - X_mean
input_image = X[0]
input_image = input_image / 255.0


input_image = np.expand_dims(input_image, axis=0)
print(input_image.shape)
#quit()
new_model = tf.keras.models.load_model("tests/classifier_model.h5")
a = new_model.predict(input_image)
print(a)
new_model = tf.keras.models.load_model("tests/classifier_model.keras")
b = new_model.predict(input_image)
print(b)

model = TFLiteModel("tests/classifier_model.tflite")
c = model.predict(input_image)
print(c)

model = TFLiteModel("tests/classifier_model_float16_quant.tflite")
d = model.predict(input_image)
print(d)