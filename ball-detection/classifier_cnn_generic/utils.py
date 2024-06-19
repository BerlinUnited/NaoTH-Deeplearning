from math import cos, pi, sin
from random import choice as random_choice

import keras
import numpy as np
import tensorflow as tf
from h5py import File
from mlflow import log_metrics, log_params, log_text
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import ExceptionSafeClass
from sklearn.metrics import classification_report, precision_recall_curve


def load_h5_dataset_X_y(file_path, rescale=None):
    with File(file_path, "r") as h5_file:
        X: np.ndarray = h5_file["X"][:]
        y: np.ndarray = h5_file["y"][:]

    if rescale is not None:
        X = X / rescale

    return X, y


@tf.function
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
    radius, x, y = target[0], target[1], target[2]

    image, x, y = random_rotation_with_coordinates(image, x, y)

    return image, (radius, x, y)


def random_wrapper(func, *args, prob=0.5):
    # with a chance of prob, apply the function to args,
    # otherwise return args unchanged

    if np.random.random() <= prob:
        return lambda *args: func(*args)
    else:
        return lambda *args: args


def augment_classification_ds(ds, prob=0.5, stddev=0.001, delta=0.1):
    flip_lr = lambda x, y: (tf.image.random_flip_left_right(x), y)
    flip_ud = lambda x, y: (tf.image.random_flip_up_down(x), y)
    rotate = lambda x, y: (random_rotation(x), y)
    brightness = lambda x, y: (random_brightness(x, delta), y)
    noise = lambda x, y: (gauss_noise(x, stddev), y)

    ds = ds.map(
        random_wrapper(flip_lr, prob=prob),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.map(
        random_wrapper(flip_ud, prob=prob),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.map(
        random_wrapper(rotate, prob=prob),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.map(
        random_wrapper(brightness, prob=prob),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.map(
        random_wrapper(noise, prob=prob),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return ds


def make_classification_dataset(
    X,
    y,
    rescale=False,
    augment=False,
    batch_size=1024,
    prob=0.5,
    stddev=0.001,
    delta=0.1,
):
    X = X.astype(np.float32)

    if rescale:
        X = rescale_func(X)

    # Create Dataset on CPU, because GPU memory is limited
    with tf.device("CPU"):
        ds = tf.data.Dataset.from_tensor_slices((X, y))

    if augment:
        ds = augment_classification_ds(ds, prob=prob, stddev=stddev, delta=delta)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds


def make_callbacks(model_name, mlflow=False):
    log_callback = keras.callbacks.TensorBoard(log_dir=f"../../data/logs/{model_name}/")
    reduce_callback = keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=20, verbose=0, mode="auto")
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=300, restore_best_weights=True)

    callbacks = [log_callback, reduce_callback, early_stopping_callback]

    if mlflow:
        callbacks.append(MLflowCallback())

    return callbacks


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_false_positives_count(y_true, y_pred):
    false_positive_count = 0

    for i in range(len(y_true)):
        if y_pred[i] == 1 and y_true[i] == 0:
            false_positive_count += 1

    return false_positive_count


def get_false_negatives_count(y_true, y_pred):
    false_negative_count = 0

    for i in range(len(y_true)):
        if y_pred[i] == 0 and y_true[i] == 1:
            false_negative_count += 1

    return false_negative_count


def get_false_positive_images(X, y_true, y_pred):
    false_positive_images = []

    for i in range(len(y_true)):
        if y_pred[i] == 1 and y_true[i] == 0:
            false_positive_images.append(X[i])

    return false_positive_images


def get_false_negative_images(X, y_true, y_pred):
    false_negative_images = []

    for i in range(len(y_true)):
        if y_pred[i] == 0 and y_true[i] == 1:
            false_negative_images.append(X[i])

    return false_negative_images


def get_classification_report_metrics(y_true, y_pred):
    clf_report: dict = classification_report(y_true, y_pred, output_dict=True, target_names=["no_ball", "ball"])

    clf_report["false_positives"] = get_false_positives_count(y_true, y_pred)
    clf_report["false_negatives"] = get_false_negatives_count(y_true, y_pred)

    return clf_report


def get_optimized_classification_report_metrics(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # Find the optimal threshold
    # precision and recall have shape (n_thresholds + 1,)
    optimal_idx_precision = np.argmax(precision)
    optimal_idx_threshold = min(optimal_idx_precision, len(thresholds) - 1)

    optimal_threshold = thresholds[optimal_idx_threshold]
    y_pred = [1 if p > optimal_threshold else 0 for p in y_prob]

    optimal_clf_report = get_classification_report_metrics(y_true, y_pred)

    return (optimal_threshold, optimal_clf_report)


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
