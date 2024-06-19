import pickle
import shutil
from pathlib import Path

import h5py
import keras
import numpy as np
import requests
import tensorflow as tf
from losses import WeightedBinaryCrossentropy
from models import build_classifier_cnn_ball_gopen24_functional
from sklearn.model_selection import train_test_split


def X_y_from_h5(file_path):
    with h5py.File(file_path, "r") as f:
        X: np.ndarray = f["X"][:]
        y: np.ndarray = f["y"][:]

    return X, y


def transform_data(X, y):
    # scale patches to [0, 1]
    X = X / 255.0

    # subtract mean to zero-center
    X_mean = X.mean()
    print("Subtracting mean from images: ", X.mean())
    X = X - X_mean

    # One-hot encode labels
    y = keras.utils.to_categorical(y, num_classes=2)

    return X, y


def make_tf_datasets(X, y, batch_size=4 * 1024):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    with tf.device("/cpu:0"):
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_ds = train_ds.shuffle(1000).batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    return train_ds, val_ds


def make_callbacks():
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=100, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=10, min_lr=1e-7),
    ]

    return callbacks


if __name__ == "__main__":
    # Load data from remote if not available locally
    X, y = X_y_from_h5("classification_patches_yuv888_devils_ball_no_ball_X_y.h5")
    X, y = transform_data(X, y)

    # Make tf.Datasets for training
    train_ds, val_ds = make_tf_datasets(X, y)

    # TODO: Add dataset augmentations after debugging
    # Try augmenting a smaller fraction of the dataset each epoch

    # TODO: Add MLflow tracking

    # Build model
    model = build_classifier_cnn_ball_gopen24_functional()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=WeightedBinaryCrossentropy(
            weights=[1.0, 10.0],
        ),
        metrics=["accuracy"],
    )
    print(model.summary())

    # Train model, maybe add tf.device("/gpu:0") context manager here
    callbacks = make_callbacks()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=2000,
        callbacks=callbacks,
    )

    model.save("classifier_cnn_ball_gopen24.keras")
    print("Model saved to classifier_cnn_ball_gopen24.keras")

    with open("classifier_cnn_ball_gopen24_history.pkl", "wb") as f:
        pickle.dump(history, f)
    print("History saved to classifier_cnn_ball_gopen24_history.pkl")

    print("Done")
