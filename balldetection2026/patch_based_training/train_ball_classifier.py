import argparse
import json
import os
import sys
import re
import sys
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LeakyReLU,
    MaxPool2D,
    ReLU,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L1L2, L2


def build_classifier_cnn_ball_gopen24_functional():
    input_shape = (16, 16, 1)
    inputs = Input(shape=input_shape)

    x = Conv2D(16, (5, 5), padding="same", name="Conv2D_1")(inputs)
    x = ReLU(name="activation_1")(x)

    x = Conv2D(
        16,
        (5, 5),
        padding="valid",
        strides=(2, 2),
        kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
        name="Conv2D_2",
    )(x)
    x = ReLU(name="activation_2")(x)

    x = Conv2D(
        16,
        (3, 3),
        padding="valid",
        kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
        name="Conv2D_3",
    )(x)
    x = ReLU(name="activation_3")(x)

    x = Conv2D(
        16,
        (3, 3),
        padding="valid",
        kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
        name="Conv2D_4",
    )(x)
    x = ReLU(name="activation_4")(x)

    x = Flatten(name="flatten_1")(x)

    x = Dense(
        256,
        activation="relu",
        kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=L2(1e-4),
    )(x)
    x = Dropout(0.1)(x)

    x = Dense(
        32,
        activation="relu",
        kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=L2(1e-4),
    )(x)

    outputs = Dense(1, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def make_naoth_detector_generic_functional(
    input_shape=(16, 16, 1),
    filters=(8, 8, 16, 16),
    n_dense=64,
    regularize=True,
    final_activation="sigmoid",
    static_batch_size=None,
):
    inputs = Input(shape=input_shape, batch_size=static_batch_size)

    x = Conv2D(filters[0], (3, 3), padding="same", name="Conv2D_1")(inputs)
    x = BatchNormalization(name="batch_norm_1")(x)
    x = LeakyReLU(name="activation_1")(x)
    x = MaxPool2D(pool_size=(2, 2), name="pooling_1")(x)

    x = Conv2D(
        filters[1],
        (3, 3),
        padding="same",
        name="Conv2D_2",
        kernel_regularizer=(L1L2(l1=1e-5, l2=1e-4) if regularize else None),
        bias_regularizer=L2(1e-4) if regularize else None,
    )(x)
    x = BatchNormalization(name="batch_norm_2")(x)
    x = LeakyReLU(name="activation_2")(x)
    x = MaxPool2D(pool_size=(2, 2), name="pooling_2")(x)

    x = Conv2D(
        filters[2],
        (3, 3),
        padding="same",
        name="Conv2D_3",
        kernel_regularizer=(L1L2(l1=1e-5, l2=1e-4) if regularize else None),
        bias_regularizer=L2(1e-4) if regularize else None,
    )(x)
    x = BatchNormalization(name="batch_norm_3")(x)
    x = LeakyReLU(name="activation_3")(x)
    x = MaxPool2D(pool_size=(2, 2), name="pooling_3")(x)

    x = Conv2D(
        filters[3],
        (2, 2),
        padding="valid",
        name="Conv2D_4",
        kernel_regularizer=(L1L2(l1=1e-5, l2=1e-4) if regularize else None),
        bias_regularizer=L2(1e-4) if regularize else None,
    )(x)
    x = BatchNormalization(name="batch_norm_4")(x)
    x = LeakyReLU(name="activation_5")(x)

    x = Flatten(name="flatten_1")(x)
    x = Dense(
        n_dense,
        activation="leaky_relu",
        kernel_regularizer=(L1L2(l1=1e-5, l2=1e-4) if regularize else None),
        bias_regularizer=L2(1e-4) if regularize else None,
    )(x)

    if regularize:
        x = Dropout(0.33)(x)

    outputs = Dense(1, activation=final_activation)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", type=str, help="Set BOTTOM or TOP")
    args = parser.parse_args()

    if args.camera is None:
        print("The camera is not set.\nSet with option -c, --camera TOP/BOTTOM")
        sys.exit()

    mlflow.set_experiment("Naoth_Ball_Detection")
    mlflow.tensorflow.autolog()

    with mlflow.start_run(run_name="32x32_patch_training"):
        mlflow.log_param("user", os.environ.get("MLFLOW_USER"))

        seed = 42

        (train_ds, val_ds) = tf.keras.utils.image_dataset_from_directory(
            f"data/{args.camera}/patches",
            validation_split=0.2,
            subset="both",
            seed=seed,
            image_size=(16, 16),
            color_mode="grayscale",
            batch_size=32,
            labels="inferred",
            label_mode="binary",
            class_names=["noball", "ball"],
        )

        val_image_names = set()
        for path in val_ds.file_paths:
            filename = Path(path).stem

            base_stem = re.sub(r"_(no)?ball_\d+$", "", filename)
            val_image_names.add(f"{base_stem}.png")

        list_path = f"data/{args.camera}/val_images.json"
        with open(list_path, "w") as f:
            json.dump(list(val_image_names), f, indent=4)

        model = make_naoth_detector_generic_functional()
        model.summary()

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        mlflow.log_param("regularization", True)

        model.fit(train_ds, validation_data=val_ds, epochs=10)

        model_path = f"data/{args.camera}/trionda_small_{args.camera}.keras"
        model.save(model_path)
