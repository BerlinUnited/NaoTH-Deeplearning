import argparse
import pickle
import time
from pathlib import Path
from pprint import pprint

import mlflow
import mlflow.data.numpy_dataset
import numpy as np
import tensorflow as tf
from losses import WeightedBinaryCrossentropy
from models import make_naoth_classifier_generic_functional
from PIL import Image as PIL_Image
from sklearn.model_selection import train_test_split
from tensorflow import keras
from utils import (
    flatten_dict,
    get_classification_report_metrics,
    get_false_negative_images,
    get_false_positive_images,
    get_optimized_classification_report_metrics,
    load_h5_dataset_X_y,
    make_callbacks,
    make_classification_dataset,
)

from tools.helper import get_file_from_server, str2bool
from tools.mflow_helper import set_tracking_url


def parse_args():
    parser = argparse.ArgumentParser(description="Train a neural network with specified parameters.")

    parser.add_argument("--mlflow_experiment", type=str, help="Name of the MLFlow experiment to log.")
    parser.add_argument("--mlflow_server", type=str, help="Server for Tracking")
    parser.add_argument(
        "--mlflow_fail_on_timeout",
        type=str2bool,
        default=False,
        help="wether to fail the training if mlflow server is unreachable",
    )
    parser.add_argument("--epochs", type=int, default=2000, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=6 * 1024, help="Batch size for training.")
    parser.add_argument(
        "--augment_training",
        type=str2bool,
        default=True,
        help="Whether to use data augmentation during training.",
    )
    parser.add_argument("--rescale", type=str2bool, default=True, help="Whether to rescale the input data.")
    parser.add_argument(
        "--subtract_mean",
        type=str2bool,
        default=True,
        help="Whether to subtract the mean from the input data.",
    )
    parser.add_argument(
        "--filters",
        type=int,
        nargs="+",
        default=[4, 4, 8, 8],
        help="List of filters for each convolutional layer.",
    )
    parser.add_argument("--regularize", type=str2bool, default=True, help="Whether to apply regularization.")
    parser.add_argument("--n_dense", type=int, default=64, help="Number of units in the dense layer.")
    parser.add_argument(
        "--input_shape",
        type=int,
        nargs=3,
        default=(16, 16, 1),
        help="Shape of the input data.",
    )
    parser.add_argument(
        "--data_root", type=str, default="../../data", help="Root directory for loading and saving data"
    )
    parser.add_argument("--data_train", type=str, help="Path to the training data.")
    parser.add_argument("--data_val", type=str, default=None, help="Path to the validation data.")
    parser.add_argument("--data_test", type=str, default=None, help="Path to the test data.")

    return parser.parse_args()


def load_data_train_test_val(
    data_train,
    data_val,
    data_test=None,
    input_shape=None,
    to_categorical=True,
    rescale=True,
    subtract_mean=True,
):
    if not input_shape:
        raise ValueError("Input shape must be provided.")

    # Load the training, validation and test datasets
    # If no validation dataset is provided, split the training dataset
    X_train, y_train = load_h5_dataset_X_y(data_train)

    if data_val:
        X_val, y_val = load_h5_dataset_X_y(data_val)
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.15,
            random_state=42,
            stratify=y_train,
        )

    X_test, y_test = load_h5_dataset_X_y(data_test) if data_test else (None, None)

    # Subtract the mean from the datasets if required
    X_train_mean = None
    if subtract_mean:
        X_train_mean = X_train.mean()
        X_train = X_train.astype(np.float32) - X_train_mean
        X_val = X_val.astype(np.float32) - X_train_mean

        if X_test is not None:
            X_test = X_test.astype(np.float32) - X_train_mean

    if rescale:
        X_train = X_train / 255.0
        X_val = X_val / 255.0
        X_test = X_test / 255.0 if X_test is not None else None

    if to_categorical:
        y_train = keras.utils.to_categorical(y_train, num_classes=2)
        y_val = keras.utils.to_categorical(y_val, num_classes=2)
        y_test = keras.utils.to_categorical(y_test, num_classes=2) if y_test is not None else None

    X_train = X_train.reshape(-1, *input_shape)
    X_val = X_val.reshape(-1, *input_shape)
    X_test = X_test.reshape(-1, *input_shape) if X_test is not None else None

    return X_train, y_train, X_val, y_val, X_test, y_test, X_train_mean


def make_model_name(args):
    date_ymd = time.strftime("%Y-%m-%d")
    loss_str = "weighted_bce_10to1"  # TODO: make this configurable
    train_data_str = args.data_train.split("/")[-1].split(".")[0]
    color = "yuv422_y_only" if args.input_shape[-1] == 1 else "yuv422"
    filters_str = "FL" + "_".join(map(str, args.filters))
    dense_str = f"D{args.n_dense}"
    dropout_str = f"DO{0.33}" if args.regularize else "noDO"
    regularize_str = "reg" if args.regularize else "noReg"
    augment_str = "aug" if args.augment_training else "noAug"
    return f"{date_ymd}_ball_classifier_{color}_{train_data_str}_{filters_str}_{dense_str}_{dropout_str}_{regularize_str}_{augment_str}_{loss_str}"


def download_data(data_path):
    if not data_path.exists():
        url = f"https://datasets.naoth.de/classification/{data_path.name}"
        print(f"Downloading {url} to {data_path}")
        get_file_from_server(url, data_path)


def save_images_as_png(images, path, X_mean=None):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    for i, image in enumerate(images):
        if X_mean:
            image = image + X_mean
        image = image * 255
        image = image.astype(np.uint8)
        image = image.squeeze()
        image = PIL_Image.fromarray(image)
        image.save(path / f"{i}.png")


def zip_image_dir(input_dir_path, output_zip_path, delete_source=True):
    import shutil

    shutil.make_archive(output_zip_path, "zip", input_dir_path)

    if delete_source:
        shutil.rmtree(input_dir_path)


if __name__ == "__main__":
    args = parse_args()

    MODEL_NAME = make_model_name(args)
    DATA_ROOT = Path(args.data_root)
    MODEL_ROOT = DATA_ROOT / "models" / MODEL_NAME

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    data_train = DATA_ROOT / f"{args.data_train}"
    data_val = DATA_ROOT / f"{args.data_val}"
    data_test = DATA_ROOT / f"{args.data_test}" if args.data_test else None

    download_data(data_train)
    download_data(data_val)
    if data_test:
        download_data(data_test)

    if args.subtract_mean and args.input_shape[-1] != 1:
        print("Subtracting mean only supported for single channel images, setting to False for color images")
        subtract_mean = False
    else:
        subtract_mean = args.subtract_mean

    X_train, y_train, X_val, y_val, X_test, y_test, X_mean = load_data_train_test_val(
        data_train,
        data_val,
        data_test,
        input_shape=args.input_shape,
        to_categorical=True,
        rescale=args.rescale,
        subtract_mean=args.subtract_mean,
    )

    train_ds = make_classification_dataset(
        X=X_train,
        y=y_train,
        batch_size=args.batch_size,
        augment=args.augment_training,
        rescale=False,
        prob=0.33,
        stddev=0.025,
        delta=0.1,
    )

    val_ds = make_classification_dataset(
        X=X_val,
        y=y_val,
        batch_size=args.batch_size,
        rescale=False,
        augment=False,
    )

    classifier = make_naoth_classifier_generic_functional(
        input_shape=args.input_shape,
        filters=args.filters,
        regularize=args.regularize,
        n_dense=args.n_dense,
    )

    classifier.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=WeightedBinaryCrossentropy(
            weights=[1.0, 10.0],
        ),
        metrics=["accuracy"],
    )

    # set up mlflow tracking
    set_tracking_url(url=args.mlflow_server, fail_on_timeout=args.mlflow_fail_on_timeout)

    experiment_tags = {
        "user": "max",
        "run_name": MODEL_NAME,
        "date": time.strftime("%Y-%m-%d"),
    }

    mlflow_experiment = mlflow.set_experiment(args.mlflow_experiment)

    with mlflow.start_run() as run:
        mlflow.set_experiment_tags(experiment_tags)

        mlflow.log_params(vars(args))

        mlflow.log_input(
            mlflow.data.from_numpy(
                np.array([]),
                source=f"https://datasets.naoth.de/classification/{args.data_train}",
                name=args.data_train,
            ),
            context="training",
        )
        mlflow.log_input(
            mlflow.data.from_numpy(
                np.array([]),
                source=f"https://datasets.naoth.de/classification/{args.data_val}",
                name=args.data_val,
            ),
            context="validation",
        )

        if args.data_test:
            mlflow.log_input(
                mlflow.data.from_numpy(
                    np.array([]),
                    source=f"https://datasets.naoth.de/classification/{args.data_test}",
                    name=args.data_test,
                ),
                context="test",
            )

        callbacks = make_callbacks(mlflow=True)

        with tf.device("/device:GPU:0"):
            history = classifier.fit(
                train_ds,
                epochs=args.epochs,
                batch_size=args.batch_size,
                validation_data=val_ds,
                callbacks=callbacks,
            )

        classifier.save(MODEL_ROOT / "classifier_model.keras")
        classifier.save(MODEL_ROOT / "classifier_model.h5")

        # save history
        with open(MODEL_ROOT / "classifier.history.pkl", "wb") as f:
            pickle.dump(history.history, f)

        mlflow.keras.log_model(
            classifier,
            artifact_path=MODEL_NAME,
            registered_model_name=MODEL_NAME,
        )

        # Create FP and FN images for training and validation data
        y_pred_train = classifier.predict(train_ds)
        y_pred_val = classifier.predict(val_ds)

        y_true_train = np.argmax(y_train, axis=1)
        y_pred_train = np.argmax(y_pred_train, axis=1)
        y_true_val = np.argmax(y_val, axis=1)
        y_pred_val = np.argmax(y_pred_val, axis=1)

        X_fp_train = get_false_positive_images(X_train, y_true_train, y_pred_train)
        X_fn_train = get_false_negative_images(X_train, y_true_train, y_pred_train)
        X_fp_val = get_false_positive_images(X_val, y_true_val, y_pred_val)
        X_fn_val = get_false_negative_images(X_val, y_true_val, y_pred_val)

        X_mean = X_mean if subtract_mean else None
        save_images_as_png(X_fp_train, MODEL_ROOT / "false_positives_train", X_mean=X_mean)
        save_images_as_png(X_fn_train, MODEL_ROOT / "false_negatives_train", X_mean=X_mean)
        save_images_as_png(X_fp_val, MODEL_ROOT / "false_positives_val", X_mean=X_mean)
        save_images_as_png(X_fn_val, MODEL_ROOT / "false_negatives_val", X_mean=X_mean)

        zip_image_dir(MODEL_ROOT / "false_positives_train", MODEL_ROOT / "false_positives_train", delete_source=True)
        zip_image_dir(MODEL_ROOT / "false_negatives_train", MODEL_ROOT / "false_negatives_train", delete_source=True)
        zip_image_dir(MODEL_ROOT / "false_positives_val", MODEL_ROOT / "false_positives_val", delete_source=True)
        zip_image_dir(MODEL_ROOT / "false_negatives_val", MODEL_ROOT / "false_negatives_val", delete_source=True)

        # TODO: save the images as artifacts

        if args.data_test:

            y_pred = classifier.predict(X_test)
            y_prob = y_pred[:, 1]
            y_pred = np.argmax(y_pred, axis=1)
            y_test = np.argmax(y_test, axis=1)

            X_fp_test = get_false_positive_images(X_test, y_test, y_pred)
            X_fn_test = get_false_negative_images(X_test, y_test, y_pred)

            save_images_as_png(X_fp_test, MODEL_ROOT / "false_positives_test", X_mean=X_mean)
            save_images_as_png(X_fn_test, MODEL_ROOT / "false_negatives_test", X_mean=X_mean)

            zip_image_dir(MODEL_ROOT / "false_positives_test", MODEL_ROOT / "false_positives_test", delete_source=True)
            zip_image_dir(MODEL_ROOT / "false_negatives_test", MODEL_ROOT / "false_negatives_test", delete_source=True)

            # Get the classification report metrics
            clf_report = get_classification_report_metrics(y_test, y_pred)

            # Get the optimized classification report metrics for maximum precision
            opt_threshold, opt_clf_report = get_optimized_classification_report_metrics(y_test, y_prob)

            report_dict = {
                "clf_report": clf_report,
                "optimal_threshold": opt_threshold,
                "optimal_clf_report": opt_clf_report,
            }

            # flatten the report dict by traversing the nested dicts
            # and concatenating the keys until a scalar value is reached
            report_dict = flatten_dict(report_dict)

            # remove empty values
            report_dict = {k: v for k, v in report_dict.items() if v}

            for key, value in report_dict.items():
                mlflow.log_metric(key, value)

            print("Test metrics:")
            pprint(report_dict)

            with open(MODEL_ROOT / "test_metrics.txt", "w") as f:
                f.write(str(report_dict))

        # Log all artifacts
        mlflow.log_artifacts(str(MODEL_ROOT))
