"""
    FIXME add tflite export
    FIXME add early stopping callback
    FIXME add saving callback
"""
import argparse
import h5py
import numpy as np
from tensorflow import keras
from keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import bhuman_segmentation_lower_gray, bhuman_segmentation_lower_rgb, bhuman_segmentation_lower_yuv, bhuman_segmentation_lower_y


class DataGenerator(Sequence):
    # FIXME dont do the augmentation for validation set
    def __init__(self, file_path, batch_size):
        self.file_path = file_path
        self.batch_size = batch_size
        with h5py.File(file_path, 'r') as f:
            self.num_samples = len(f['X'])
        
        self.datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        with h5py.File(self.file_path, 'r') as f:
            start_index = index * self.batch_size
            end_index = min((index + 1) * self.batch_size, self.num_samples)
            X_batch = f['X'][start_index:end_index]
            y_batch = f['Y'][start_index:end_index]
        augmented = next(self.datagen.flow(X_batch, y_batch, batch_size=self.batch_size))
        images, masks = augmented[0], augmented[1]
        return images, masks


def train_rgb():
    train_generator = DataGenerator('training_ds_rgb.h5', batch_size=32)
    validation_generator = DataGenerator('validation_ds_rgb.h5', batch_size=32)

    model = bhuman_segmentation_lower_rgb()
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['accuracy'])

    model.fit(x=train_generator, validation_data=validation_generator, epochs=200)
    model.save(f"semantic_segmentation_rgb.keras")


def train_grayscale_stupid():
    train_generator = DataGenerator('training_ds_gray.h5', batch_size=32)
    validation_generator = DataGenerator('validation_ds_gray.h5', batch_size=32)

    model = bhuman_segmentation_lower_gray()
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['accuracy'])

    model.fit(x=train_generator, validation_data=validation_generator, epochs=200)
    model.save(f"semantic_segmentation_gray.keras")


def train_grayscale_y():
    train_generator = DataGenerator('training_ds_y.h5', batch_size=32)
    validation_generator = DataGenerator('validation_ds_y.h5', batch_size=32)

    model = bhuman_segmentation_lower_y()
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['accuracy'])

    callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=50, restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.75, patience=10),
]

    model.fit(x=train_generator, validation_data=validation_generator, callbacks=callbacks, epochs=500, shuffle=True)
    model.save(f"semantic_segmentation_y-2.keras")

def train_yuv422():
    train_generator = DataGenerator('training_ds_yuv.h5', batch_size=32)
    validation_generator = DataGenerator('validation_ds_yuv.h5', batch_size=32)

    model = bhuman_segmentation_lower_yuv()
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['accuracy'])

    model.fit(x=train_generator, validation_data=validation_generator, epochs=200)
    model.save(f"semantic_segmentation_yuv.keras")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", required=True, choices=['gray', 'yuv', 'rgb', 'y'])
    args = parser.parse_args()

    if args.type == "yuv":
        train_yuv422()
    if args.type == "gray":
        train_grayscale_stupid()
    if args.type == "rgb":
        train_rgb()
    if args.type == "y":
        train_grayscale_y()