from math import cos, pi, sin
from random import choice as random_choice

import numpy as np
import tensorflow as tf
from h5py import File
from matplotlib import pyplot as plt
from mlflow import log_metrics, log_params, log_text
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import ExceptionSafeClass
from tensorflow import keras


def plot_images_with_ball_center_and_radius(X, y, save_name=None, show=False):
    save_name = save_name or "ball_images_with_center_and_radius.png"

    width = height = int(np.sqrt(X.shape[1]))

    fig, ax = plt.subplots(height, width, figsize=(20, 20))

    for i in range(height):
        for j in range(width):
            index = i * width + j
            ball_image = X[index]
            target = y[index]

            ball_x, ball_y, ball_radius = target

            ax[i][j].imshow(ball_image, cmap="gray")

            # scale x, y. radius to image size
            ball_x = ball_x * ball_image.shape[1]
            ball_y = ball_y * ball_image.shape[0]
            ball_radius = ball_radius * ball_image.shape[1]

            # draw x, y as ball center into the images with radius
            ax[i][j].scatter(ball_x, ball_y, c="r", s=10)
            ax[i][j].add_patch(plt.Circle((ball_x, ball_y), ball_radius, color="g", fill=True, alpha=0.2))

            ax[i][j].axis("off")

    if show:
        plt.show()

    fig.savefig(save_name)


def load_h5_dataset_X_y(file_path, rescale=None):
    with File(file_path, "r") as h5_file:
        X: np.ndarray = h5_file["X"][:]
        y: np.ndarray = h5_file["y"][:]

    if rescale is not None:
        X = X / rescale

    return X, y


def rescale_func(X, factor=255.0):
    return X / factor


@tf.function
def gauss_noise(image, stddev=0.001):
    return image + tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=stddev, dtype=tf.float32)


@tf.function
def random_brightness(image, delta=0.3):
    image = tf.image.random_brightness(image, delta)

    return image


@tf.function
def random_rotation(image):
    k = random_choice([0, 1, 2, 3])
    return tf.image.rot90(image, k=k)


@tf.function
def random_rotation_with_coordinates(image, x, y, k=None):
    if k is None:
        k = random_choice([0, 1, 2, 3])

    height, width = image.shape[:2]

    xm, ym = width // 2, height // 2
    x = x * width
    y = y * height

    a = k * -90 * pi / 180

    x_new = (x - xm) * cos(a) - (y - ym) * sin(a) + xm
    y_new = (x - xm) * sin(a) + (y - ym) * cos(a) + ym

    x_new = x_new / width
    y_new = y_new / height

    return tf.image.rot90(image, k=k), x_new, y_new


@tf.function
def random_rotation_with_coordinates_ds(image, target):
    x, y, radius = target[0], target[1], target[2]

    image, x, y = random_rotation_with_coordinates(image, x, y)

    return image, (x, y, radius)


def random_wrapper(func, *args, prob=0.5):
    # with a chance of prob, apply the function to args,
    # otherwise return args unchanged

    if np.random.random() <= prob:
        return lambda *args: func(*args)
    else:
        return lambda *args: args


def make_detection_dataset(X, y, batch_size, augment=False, rescale=True, prob=0.5, stddev=0.1, delta=0.1):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    rotate = lambda x, y: (random_rotation_with_coordinates_ds(x, y))
    brightness = lambda x, y: (random_brightness(x, delta), y)
    noise = lambda x, y: (gauss_noise(x, stddev), y)

    if augment:
        # dataset = dataset.map(
        #     random_wrapper(rotate, prob=prob),
        #     num_parallel_calls=tf.data.experimental.AUTOTUNE,
        # )

        dataset = dataset.map(
            random_wrapper(brightness, prob=prob),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        dataset = dataset.map(
            random_wrapper(noise, prob=prob),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    if rescale:
        dataset = dataset.map(
            lambda x, y: (rescale_func(x), y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def make_callbacks(mlflow=False):
    reduce_callback = keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=20, verbose=0, mode="auto")
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=300, restore_best_weights=True)

    callbacks = [reduce_callback, early_stopping_callback]

    if mlflow:
        callbacks.append(MLflowCallback())

    return callbacks


# I'm having trouble importing this in docker, hack it in here
@experimental
class MLflowCallback(keras.callbacks.Callback, metaclass=ExceptionSafeClass):

    def __init__(self, log_every_epoch=True, log_every_n_steps=None):
        self.log_every_epoch = log_every_epoch
        self.log_every_n_steps = log_every_n_steps

        if log_every_epoch and log_every_n_steps is not None:
            raise ValueError(
                "`log_every_n_steps` must be None if `log_every_epoch=True`, received "
                f"`log_every_epoch={log_every_epoch}` and `log_every_n_steps={log_every_n_steps}`."
            )

        if not log_every_epoch and log_every_n_steps is None:
            raise ValueError(
                "`log_every_n_steps` must be specified if `log_every_epoch=False`, received"
                "`log_every_n_steps=False` and `log_every_n_steps=None`."
            )

    def on_train_begin(self, logs=None):
        """Log model architecture and optimizer configuration when training begins."""
        config = self.model.optimizer.get_config()
        log_params({f"optimizer_{k}": v for k, v in config.items()})

        model_summary = []

        def print_fn(line, *args, **kwargs):
            model_summary.append(line)

        self.model.summary(print_fn=print_fn)
        summary = "\n".join(model_summary)
        log_text(summary, artifact_file="model_summary.txt")

    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at the end of each epoch."""
        if not self.log_every_epoch or logs is None:
            return
        log_metrics(logs, step=epoch, synchronous=False)

    def on_batch_end(self, batch, logs=None):
        """Log metrics at the end of each batch with user specified frequency."""
        if self.log_every_n_steps is None or logs is None:
            return
        current_iteration = int(self.model.optimizer.iterations.numpy())

        if current_iteration % self.log_every_n_steps == 0:
            log_metrics(logs, step=current_iteration, synchronous=False)

    def on_test_end(self, logs=None):
        """Log validation metrics at validation end."""
        if logs is None:
            return
        metrics = {"validation_" + k: v for k, v in logs.items()}
        log_metrics(metrics, synchronous=False)
