import pickle
import shutil
from pathlib import Path

import h5py
import keras
import numpy as np
import requests
import tensorflow as tf
from sklearn.model_selection import train_test_split


def X_y_from_h5(file_path):
    with h5py.File(file_path, "r") as f:
        X: np.ndarray = f["X"][:]
        y: np.ndarray = f["y"][:]

    return X, y


def get_or_load_data():
    # TODO use a function from tools/helper.py
    """
    Try to load the data from the local file. If the file does not exist,
    download it from the remote URL.

    Returns X, y
    """
    remote_url = "https://datasets.naoth.de/NaoDevils_Patches_GO24_32x32x3_nsamples_%20215820/patches_classification_naodevils_ball_no_ball_X_y.h5"

    local_base = Path(__file__).parent 
    dataset_name = (
        "classification_gopen24_nao_devils_labelstudio_validated_ball_no_ball_X_y.h5"
    )
    local_file = local_base / dataset_name

    if not local_file.exists():
        print(f"Downloading dataset from {remote_url} to {local_file}")

        try:
            with requests.get(remote_url, stream=True) as r, open(
                local_file, "wb"
            ) as f:
                shutil.copyfileobj(r.raw, f)
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            raise
    else:
        print(f"Dataset already exists at {local_file}, loading locally...")

    X, y = X_y_from_h5(local_file)
    print(f"Dataset shape: X {X.shape}, y {y.shape}")

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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    with tf.device("/cpu:0"):
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_ds = train_ds.shuffle(1000).batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    return train_ds, val_ds


def build_classifier_cnn_ball_gopen24():
    input_shape = (16, 16, 1)
    classifier = keras.models.Sequential()

    classifier.add(
        keras.layers.Convolution2D(
            16, (5, 5), input_shape=input_shape, padding="same", name="Conv2D_1"
        )
    )
    classifier.add(keras.layers.ReLU(name="activation_1"))

    classifier.add(
        keras.layers.Convolution2D(
            16,
            (5, 5),
            padding="valid",
            name="Conv2D_2",
            strides=(2, 2),
            kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
        )
    )
    classifier.add(keras.layers.ReLU(name="activation_2"))

    classifier.add(
        keras.layers.Convolution2D(
            16,
            (3, 3),
            padding="valid",
            name="Conv2D_3",
            kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
        )
    )

    classifier.add(keras.layers.ReLU(name="activation_3"))

    classifier.add(
        keras.layers.Convolution2D(
            16,
            (3, 3),
            padding="valid",
            name="Conv2D_4",
            kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
        )
    )

    classifier.add(keras.layers.ReLU(name="activation_4"))

    classifier.add(keras.layers.Flatten(name="flatten_1"))

    classifier.add(
        keras.layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
            bias_regularizer=keras.regularizers.L2(1e-4),
        )
    )
    classifier.add(keras.layers.Dropout(0.1))
    classifier.add(
        keras.layers.Dense(
            32,
            activation="relu",
            kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
            bias_regularizer=keras.regularizers.L2(1e-4),
        )
    )
    classifier.add(
        keras.layers.Dense(2, activation="softmax"),
    )

    return classifier


def make_callbacks():
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=100, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=10, min_lr=1e-7
        ),
    ]

    return callbacks


@keras.saving.register_keras_serializable(name="weighted_binary_crossentropy")
def weighted_binary_crossentropy(target, output, weights):
    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)
    weights = tf.convert_to_tensor(weights, dtype=target.dtype)

    epsilon_ = tf.constant(keras.backend.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)

    # Compute cross entropy from probabilities.
    bce = weights[1] * target * tf.math.log(output + epsilon_)
    bce += weights[0] * (1 - target) * tf.math.log(1 - output + epsilon_)
    return -bce


@keras.saving.register_keras_serializable(name="WeightedBinaryCrossentropy")
class WeightedBinaryCrossentropy:
    def __init__(
        self,
        label_smoothing=0.0,
        weights=[1.0, 1.0],
        axis=-1,
        name="weighted_binary_crossentropy",
        loss_fn=weighted_binary_crossentropy,
    ):
        """Initializes `WeightedBinaryCrossentropy` instance.

        Args:
          from_logits: Whether to interpret `y_pred` as a tensor of
            [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
            assume that `y_pred` contains probabilities (i.e., values in [0,
            1]).

          TODO: Check if this might be helpful?
          label_smoothing: Float in [0, 1]. When 0, no smoothing occurs. When >
            0, we compute the loss between the predicted labels and a smoothed
            version of the true labels, where the smoothing squeezes the labels
            towards 0.5.  Larger values of `label_smoothing` correspond to
            heavier smoothing.

          axis: The axis along which to compute crossentropy (the features
            axis).  Defaults to -1.
          name: Name for the op. Defaults to 'weighted_binary_crossentropy'.
        """
        super().__init__()
        self.weights = weights  # tf.convert_to_tensor(weights)
        self.label_smoothing = label_smoothing
        self.name = name
        self.loss_fn = weighted_binary_crossentropy if loss_fn is None else loss_fn

    def __call__(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        self.label_smoothing = tf.convert_to_tensor(
            self.label_smoothing, dtype=y_pred.dtype
        )

        def _smooth_labels():
            return y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        y_true = tf.__internal__.smart_cond.smart_cond(
            self.label_smoothing, _smooth_labels, lambda: y_true
        )

        return tf.reduce_mean(self.loss_fn(y_true, y_pred, self.weights), axis=-1)

    def get_config(self):
        config = {"name": self.name, "weights": self.weights, "loss_fn": self.loss_fn}

        return dict(list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == "__main__":
    # Load data from remote if not available locally
    X, y = get_or_load_data()
    X, y = transform_data(X, y)

    # Make tf.Datasets for training
    train_ds, val_ds = make_tf_datasets(X, y)

    # TODO: Add dataset augmentations after debugging
    # Try augmenting a smaller fraction of the dataset each epoch

    # TODO: Add MLflow tracking

    # Build model
    model = build_classifier_cnn_ball_gopen24()
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
