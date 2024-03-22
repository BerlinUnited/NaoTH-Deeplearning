# get directory of this file
import os
import pickle

import numpy as np
import tensorflow as tf
from h5py import File


def get_data_dir(data_folder="data"):
    current_dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(current_dir_path, data_folder)


def load_ds_ball_detection_center_radius(ds_name="tk03_natural_detection.pkl"):
    """
    Labeled Dataset of patches with ball center and radius.
    39308 total patches, with 19654 balls and 19654 "not balls".

    Images are 16x16x1 grayscale with values in [-1, 1].
    Target vector y is a 4-tuple: (radius, x_coord, y_coord, is_ball_bool)
    """
    file_path = os.path.join(get_data_dir(), ds_name)

    with open(file_path, "rb") as f:
        mean = pickle.load(f)
        images = pickle.load(f)
        y = pickle.load(f)  # radius, x_coord, y_coord, is_ball
        filepaths = pickle.load(f)

    return mean, images, y, filepaths


def load_ds_patches_21_23(ds_name="patches_21_23.h5"):
    """
    Unlabelled dataset of patches from the top and bottom cameras.
    Roughly 740k patches from logs between 2021 and 2023.

    Images are 16x16x1 grayscale with values in [0, 255].
    """

    file_path = os.path.join(get_data_dir(), ds_name)

    with File(file_path, "r") as h5_file:
        X_top: np.ndarray = h5_file["top"][:]
        X_bottom: np.ndarray = h5_file["bottom"][:]

    return np.concatenate([X_top, X_bottom])


def load_ds_patches_classification_ball_no_ball(
    ds_name="classification_ball_no_ball.h5",
):
    """
    Labelled dataset of patches with balls and without balls.
    789455 total patches, with 47937 balls and 741518 "not balls".

    Checked most of the ball patches manually, the quality should be good.
    No information about which camera the patches are from, use the
    "classification_ball_no_ball_top_bottom" dataset for that.

    This dataset is a combination of tk03_natural_detection, tk03_natural_classification
    and patches_21_23 datasets.

    Images are 16x16x1 grayscale with values in [0, 255].
    Target vector y is an array of 0s and 1s, where 1 means ball.
    """

    file_path = os.path.join(get_data_dir(), ds_name)

    with File(file_path, "r") as h5_file:
        X_ball: np.ndarray = h5_file["ball"][:]
        X_no_ball: np.ndarray = h5_file["no_ball"][:]

    y_ball = np.ones(X_ball.shape[0])
    y_no_ball = np.zeros(X_no_ball.shape[0])

    X = np.concatenate([X_ball, X_no_ball])
    y = np.concatenate([y_ball, y_no_ball])

    return X, y


def load_ds_patches_classification_ball_no_ball_top_bottom(
    ds_name="classification_ball_no_ball_top_bottom.h5",
):
    """
    Labelled dataset of patches with balls and without balls,
    from the top and bottom cameras.

    740229 total patches from "patches_21_23" dataset.

    555108 top images, with 9947 balls and 545161 "not balls"
    185121 bottom images with 13412 balls and 171709 "not balls"

    Images are 16x16x1 grayscale with values in [0, 255].
    Target vectors y are arrays of 0s and 1s, where 1 means ball.
    """

    file_path = os.path.join(get_data_dir(), ds_name)

    with File(file_path, "r") as h5_file:
        ball_top: np.ndarray = h5_file["top_ball"][:]
        no_ball_top: np.ndarray = h5_file["top_no_ball"][:]

        ball_bottom: np.ndarray = h5_file["bottom_ball"][:]
        no_ball_bottom: np.ndarray = h5_file["bottom_no_ball"][:]

    X_top = np.concatenate([ball_top, no_ball_top])
    y_top = np.concatenate([np.ones(ball_top.shape[0]), np.zeros(no_ball_top.shape[0])])

    X_bottom = np.concatenate([ball_bottom, no_ball_bottom])
    y_bottom = np.concatenate(
        [np.ones(ball_bottom.shape[0]), np.zeros(no_ball_bottom.shape[0])]
    )

    return X_top, y_top, X_bottom, y_bottom


def rescale_func(X, factor=255.0):
    return X / factor


@tf.function
def gauss_noise(image):
    return image + tf.random.normal(
        shape=tf.shape(image), mean=0.0, stddev=0.001, dtype=tf.float32
    )


def augment_autoencoder_ds(ds):
    ds = ds.map(tf.image.random_flip_left_right, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(tf.image.random_flip_up_down, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(gauss_noise, num_parallel_calls=tf.data.AUTOTUNE)

    return ds


def augment_classification_ds(ds):
    ds = ds.map(
        lambda x, y: (tf.image.random_flip_left_right(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.map(
        lambda x, y: (tf.image.random_flip_up_down(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.map(
        lambda x, y: (gauss_noise(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return ds


def make_autoencoder_dataset(X, rescale=True, augment=True, batch_size=1024):
    X = X.astype(np.float32)

    if rescale:
        X = rescale_func(X)

    # Create Dataset on CPU, because GPU memory is limited
    with tf.device("CPU"):
        ds = tf.data.Dataset.from_tensor_slices(X)

    if augment:
        ds = augment_autoencoder_ds(ds)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds


def make_classification_dataset(X, y, rescale=True, augment=True, batch_size=1024):
    X = X.astype(np.float32)

    if rescale:
        X = rescale_func(X)

    # Create Dataset on CPU, because GPU memory is limited
    with tf.device("CPU"):
        ds = tf.data.Dataset.from_tensor_slices((X, y))

    if augment:
        ds = augment_classification_ds(ds)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds
