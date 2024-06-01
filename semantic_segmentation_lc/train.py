import h5py
import numpy as np
from tensorflow import keras
from keras.utils import Sequence
from model import bhuman_segmentation_lower


class DataGenerator(Sequence):
    def __init__(self, file_path, batch_size):
        self.file_path = file_path
        self.batch_size = batch_size
        with h5py.File(file_path, 'r') as f:
            self.num_samples = len(f['X'])  # Assuming 'data' is the dataset name

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        with h5py.File(self.file_path, 'r') as f:
            start_index = index * self.batch_size
            end_index = min((index + 1) * self.batch_size, self.num_samples)
            X_batch = f['X'][start_index:end_index]  # Assuming 'data' is the dataset name
            y_batch = f['Y'][start_index:end_index]  # Assuming 'labels' is the target dataset name
        return X_batch, y_batch

"""
with h5py.File("semantic_ds2.h5", "r") as f:
    # argh this crashes everything again
    X_train = f["X"][:]
    Y_train = f["Y"][:]
"""

train_generator = DataGenerator('semantic_ds2.h5', batch_size=32)
#validation_generator = DataGenerator('validation_data.h5', batch_size=32)

model = bhuman_segmentation_lower()
#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

model.fit_generator(generator=train_generator, epochs=1) # use fit because fit_generator is deprecated

#history = model.fit(X_train, Y_train, epochs=1, validation_split=0.1,verbose=1)

model.save(f"segment_test.keras")
