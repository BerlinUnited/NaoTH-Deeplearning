"""
    FIXME add tflite export
    FIXME add early stopping callback
    FIXME add saving callback
"""
import sys
import os

helper_path = os.path.join(os.path.dirname(__file__), '../tools')
sys.path.append(helper_path)

import argparse
import h5py
import numpy as np
import mlflow
from tensorflow import keras
from keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import bhuman_segmentation_lower_yuv, bhuman_segmentation_y_channel_v1, bhuman_segmentation_y_channel_v2
from mflow_helper import set_tracking_url

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


class DataGeneratorAugmentation(Sequence):
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


def train_y_channel_v1(args):
    with mlflow.start_run() as run:
        run = mlflow.active_run()
        # FIXME put validation data in the same h5 file for easier processing
        train_generator = DataGenerator('training_ds_y.h5', batch_size=32)
        validation_generator = DataGenerator('validation_ds_y.h5', batch_size=32)

        dummy_dataset = mlflow.data.from_numpy(np.array([]), targets=np.array([]), source=f"https://datasets.naoth.de/{args.dataset}", name=args.dataset)
        mlflow.log_input(dummy_dataset, context="training", tags={"name": args.dataset})

        
        model = bhuman_segmentation_y_channel_v1()
        #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['accuracy'])

        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.75, patience=10),
            keras.callbacks.ModelCheckpoint(filepath=f"models/{run.info.run_name}/best.keras", monitor='loss', verbose=1, save_best_only=True, mode='auto')
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
    # FIXME add mlflow integration and upload to models.naoth.de
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", required=True, choices=['yuv', 'y'])
    parser.add_argument("-c", "--camera", required=True, choices=['bottom', 'top'])
    parser.add_argument("-ds", "--dataset", required=True)
    parser.add_argument("-u", "--user", required=True)
    args = parser.parse_args()

    os.environ["LOGNAME"] = args.user # needed because for now the docker container runs as root user
    # set up remote tracking if the mlflow tracking server is available
    set_tracking_url()
    mlflow.enable_system_metrics_logging()

    if args.type == "yuv":
        train_yuv422()
    if args.type == "y":
        mlflow.set_experiment(f"Segmentation 240x320 Y-Channel - {args.camera.capitalize()}")
        
        train_y_channel_v1(args)