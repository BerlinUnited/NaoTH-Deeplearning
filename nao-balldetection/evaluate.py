"""Evaluate the model"""
import tensorflow as tf
from pathlib import Path
import pickle
from train  import WeightedBinaryCrossentropy
from tensorflow import keras as keras
from loader import subtract_mean
DATA_DIR = Path(Path(__file__).parent.absolute() / "data").resolve()

model = tf.keras.models.load_model('rc26_classification_color_32_naodevils_training.h5')

model.summary()

data_file = str(DATA_DIR / "naodevils_training.pkl")
with open(data_file, "rb") as f:
    mean = pickle.load(f)
    pickle.load(f)  # skip input images
    pickle.load(f)  # skip trainings target: [r, x,y,1]

data_file = str(DATA_DIR / "naodevils_validation.pkl")
with open(data_file, "rb") as f:
    pickle.load(f)  # skip mean
    val_x = pickle.load(f)  # x are all input images
    val_y = pickle.load(f)  # y are the trainings target: [r, x,y,1]

val_x = subtract_mean(val_x, mean)
val_y_one_hot = keras.utils.to_categorical(val_y, num_classes=2)


results  = model.evaluate(val_x, val_y_one_hot, verbose=1)
print("Test Loss, Test Accuracy:", results)

data_file = str(DATA_DIR / "go26_patches.pkl")
with open(data_file, "rb") as f:
    pickle.load(f)  # skip mean
    val_x = pickle.load(f)  # x are all input images
    val_y = pickle.load(f)  # y are the trainings target: [r, x,y,1]

val_x = subtract_mean(val_x, mean)
val_y_one_hot = keras.utils.to_categorical(val_y, num_classes=2)


results  = model.evaluate(val_x, val_y_one_hot, verbose=1)
print("Test Loss, Test Accuracy:", results)