from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import h5py
import numpy as np
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, file_path, batch_size):
        self.file_path = file_path
        self.batch_size = batch_size
        with h5py.File(file_path, 'r') as f:
            self.num_samples = len(f['X'])

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        with h5py.File(self.file_path, 'r') as f:
            start_index = index * self.batch_size
            end_index = min((index + 1) * self.batch_size, self.num_samples)
            X_batch = f['X'][start_index:end_index] 
            y_batch = f['Y'][start_index:end_index]
        return X_batch, y_batch


def calculate_false_positives(model, h5_file):
    false_positive = 0
    with h5py.File(h5_file,'r') as h5f:
        for idx, image in enumerate(h5f['X']):
            if h5f['Y'][idx] == 0:
                continue
            image_input = np.expand_dims(image, axis=0)
            result = model.predict(image_input,verbose=0)
            if result < 0.9:
                false_positive += 1
    return false_positive

def calculate_false_negatives(model, h5_file):
    false_negative = 0
    with h5py.File(h5_file,'r') as h5f:
        for idx, image in enumerate(h5f['X']):
            if h5f['Y'][idx] == 1:
                continue
            image_input = np.expand_dims(image, axis=0)
            result = model.predict(image_input,verbose=0)
            if result >= 0.9:
                false_negative += 1
    return false_negative

print("fy1500 original data without mean subtraction")
model = tf.keras.models.load_model('fy1500_originaldata_y.keras')
train_generator = DataGenerator('fy1500_originaldata_y.h5', batch_size=32)
model.evaluate(train_generator) # 154 false positives
print("154 false positives")
print("2 false negatives")
#print(f"false positives: {calculate_false_positives(model, 'fy1500_originaldata_y.h5')}")
#print(f"false negatives: {calculate_false_negatives(model, 'fy1500_originaldata_y.h5')}")

print("fy1500 original data with mean subtraction")
model = tf.keras.models.load_model('fy1500_originaldata_y_mean_subtracted.keras')
train_generator = DataGenerator('fy1500_originaldata_y_mean_subtracted.h5', batch_size=32)
model.evaluate(train_generator)
print("82 false positives")
print("0 false negatives")
#print(f"false positives: {calculate_false_positives(model, 'fy1500_originaldata_y_mean_subtracted.h5')}")
#print(f"false negatives: {calculate_false_negatives(model, 'fy1500_originaldata_y_mean_subtracted.h5')}")

print("fy1500 original data without mean subtraction with additional data")
model = tf.keras.models.load_model('fy1500_originaldata_y_meta_info.keras')
train_generator = DataGenerator('fy1500_originaldata_y_meta_info.h5', batch_size=32)
model.evaluate(train_generator)
print("88 false positives")
print("0 false negatives")
#print(f"false positives: {calculate_false_positives(model, 'fy1500_originaldata_y_meta_info.h5')}")
#print(f"false negatives: {calculate_false_negatives(model, 'fy1500_originaldata_y_meta_info.h5')}")

print("fy1500 original data with mean subtraction with additional data")
model = tf.keras.models.load_model('fy1500_originaldata_y_mean_subtracted_meta_info.keras')
train_generator = DataGenerator('fy1500_originaldata_y_mean_subtracted_meta_info.h5', batch_size=32)
model.evaluate(train_generator)
print("137 false positives")
print("0 false negatives")
#print(f"false positives: {calculate_false_positives(model, 'fy1500_originaldata_y_mean_subtracted_meta_info.h5')}")
#print(f"false negatives: {calculate_false_negatives(model, 'fy1500_originaldata_y_mean_subtracted_meta_info.h5')}")





print("fy1500 on segmentation without mean subtraction")
model = tf.keras.models.load_model('fy1500_segmentationdata_y.keras')
train_generator = DataGenerator('fy1500_segmentationdata_y.h5', batch_size=32)
model.evaluate(train_generator)
print("99 false positives")
print("12 false negatives")
#print(f"false positives: {calculate_false_positives(model, 'fy1500_segmentationdata_y.h5')}")
#print(f"false negatives: {calculate_false_negatives(model, 'fy1500_segmentationdata_y.h5')}")

print("fy1500 on segmentation with mean subtraction")
model = tf.keras.models.load_model('fy1500_segmentationdata_y_mean_subtracted.keras')
train_generator = DataGenerator('fy1500_segmentationdata_y_mean_subtracted.h5', batch_size=32)
model.evaluate(train_generator)
print("84 false positives")
print("7 false negatives")
#print(f"false positives: {calculate_false_positives(model, 'fy1500_segmentationdata_y_mean_subtracted.h5')}")
#print(f"false negatives: {calculate_false_negatives(model, 'fy1500_segmentationdata_y_mean_subtracted.h5')}")


print("fy1500 on segmentation output without mean subtraction with additional data")
model = tf.keras.models.load_model('fy1500_segmentationdata_y_meta_info.keras')
train_generator = DataGenerator('fy1500_segmentationdata_y_meta_info.h5', batch_size=32)
model.evaluate(train_generator)
print("34 false positives")
print("1 false negatives")
#print(f"false positives: {calculate_false_positives(model, 'fy1500_segmentationdata_y_meta_info.h5')}")
#print(f"false negatives: {calculate_false_negatives(model, 'fy1500_segmentationdata_y_meta_info.h5')}")


print("fy1500 on segmentation output with mean subtraction with additional data")
model = tf.keras.models.load_model('fy1500_segmentationdata_y_mean_subtracted_meta_info.keras')
train_generator = DataGenerator('fy1500_segmentationdata_y_mean_subtracted_meta_info.h5', batch_size=32)
model.evaluate(train_generator)
print("23 false positives")
print("0 false negatives")
#print(f"false positives: {calculate_false_positives(model, 'fy1500_segmentationdata_y_mean_subtracted_meta_info.h5')}")
#print(f"false negatives: {calculate_false_negatives(model, 'fy1500_segmentationdata_y_mean_subtracted_meta_info.h5')}")
