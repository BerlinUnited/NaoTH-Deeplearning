import time
from pathlib import Path

from sklearn.model_selection import train_test_split

from patch_detection.classification.ball_classification import (
    load_encoder,
    make_classifier_mean,
    make_naoth_classifier,
    train_classifier_early_stop,
)
from patch_detection.datasets import (
    load_ds_patches_classification_ball_no_ball,
    make_classification_dataset,
)

if __name__ == "__main__":
    # TODO: make parameters configurable

    EPOCHS = 250
    BATCH_SIZE = 512

    class_weights = {
        0: 1.0,
        1: 5.0,  # ball class can be weighted higher in loss
    }

    output_path = Path("../models")

    # load the dataset
    X, y = load_ds_patches_classification_ball_no_ball()

    # split into train and validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # make tf.Dataset() objects for faster training
    X_train = make_classification_dataset(X_train, y_train, rescale=True, augment=True)
    X_val = make_classification_dataset(X_val, y_val, rescale=True, augment=False)

    timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    # use a trained autoencoder, and freeze the encoder part to make a classifier
    raise NotImplementedError("Add your trained Autoencoder model name here.")
    model_name = "XXX"

    encoder = load_encoder(f"../models/{model_name}.keras")
    classifier = make_classifier_mean(encoder)

    train_classifier_early_stop(
        model_name=f"{model_name}_classifier_full_clf_ball_no_ball",
        model=classifier,
        output_path=output_path,
        train_ds=X_train,
        validation_data=X_val,
        epochs=EPOCHS,
        class_weights=class_weights,
        timestamp_iso=timestamp_iso,
    )

    # make naoth classifier for comparison
    train_classifier_early_stop(
        model_name="naoth_classifier_full_clf_ball_no_ball",
        model=make_naoth_classifier(),
        output_path=output_path,
        train_ds=X_train,
        validation_data=X_val,
        epochs=EPOCHS,
        class_weights=class_weights,
        timestamp_iso=timestamp_iso,
    )
