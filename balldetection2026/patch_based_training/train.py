from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, LeakyReLU, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L1L2, L2
import tensorflow as tf
import argparse
import mlflow
import os

def make_naoth_detector_generic_functional(
    input_shape=(16, 16, 1),
    filters=(8, 8, 16, 16),
    n_dense=64,
    regularize=True,
    final_activation="sigmoid",
):
    inputs = Input(shape=input_shape)

    # Conv-LReLU-Pool Block 1
    x = Conv2D(filters[0], (3, 3), padding="same", name="Conv2D_1")(inputs)
    x = BatchNormalization(name="batch_norm_1")(x)
    x = LeakyReLU(name="activation_1")(x)
    x = MaxPool2D(pool_size=(2, 2), name="pooling_1")(x)

    # Conv-LReLU-Pool Block 2
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

    # Conv-LReLU-Pool Block 3
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

    # Conv-LReLU 2x2
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

    # Flatten and Dense Layers
    x = Flatten(name="flatten_1")(x)
    x = Dense(
        n_dense,
        activation="leaky_relu",
        kernel_regularizer=(L1L2(l1=1e-5, l2=1e-4) if regularize else None),
        bias_regularizer=L2(1e-4) if regularize else None,
    )(x)

    if regularize:
        x = Dropout(0.33)(x)

    # ball_x, ball_y, ball_radius
    outputs = Dense(1, activation=final_activation)(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", type=str, help="Set BOTTOM or TOP")
    args = parser.parse_args()

    # 2. Training Setup with MLflow
    mlflow.set_experiment("Naoth_Ball_Detection")

    # Enable autologging: captures metrics, parameters, and the model itself
    mlflow.tensorflow.autolog()

    with mlflow.start_run(run_name="32x32_patch_training"):
        # Load your data (assuming you use image_dataset_from_directory)
        # Note: You'll need to prepare your labels to match the 3-unit output
        mlflow.log_param("user", os.environ.get("MLFLOW_USER"))
        train_ds = tf.keras.utils.image_dataset_from_directory(
            f"data/{args.camera}/patches",
            image_size=(32, 32),
            batch_size=32,
            label_mode=None # Adjust based on your specific ground truth
        )

        model = make_naoth_detector_generic_functional()
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        # Log additional parameters manually if needed
        mlflow.log_param("regularization", True)

        model.fit(train_ds, epochs=10)