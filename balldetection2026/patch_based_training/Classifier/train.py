import argparse
import json
import os
import sys
import re
import sys
from pathlib import Path

import mlflow
import tensorflow

from model import build_classifier_cnn_ball_gopen24_functional

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

        (train_ds, val_ds) = tensorflow.keras.utils.image_dataset_from_directory(
            f"data/{args.camera}/patches",
            validation_split=0.2,
            subset="both",
            seed=seed,
            image_size=(16, 16),
            color_mode="grayscale",
            batch_size=32,
            labels="inferred",
            label_mode="categorical",
            class_names=["noball", "ball"],
        )
        # for sample in train_ds: 
        #     print(train_ds[0][0])
        val_image_names = set()
        for path in val_ds.file_paths:
            filename = Path(path).stem

            base_stem = re.sub(r"_(no)?ball_\d+$", "", filename)
            val_image_names.add(f"{base_stem}.png")

        list_path = f"data/{args.camera}/val_images.json"
        with open(list_path, "w") as f:
            json.dump(list(val_image_names), f, indent=4)

        num_train_images = len(train_ds.file_paths)
        num_val_images = len(val_ds.file_paths)

        print(f"Training images (original): {num_train_images}")
        print(f"Validation images: {num_val_images}")
        
        # normalization = tensorflow.keras.layers.Rescaling(1./255)

        # train_ds = train_ds.map(
        #     lambda x, y: (normalization(x), y),
        #     num_parallel_calls=tensorflow.data.AUTOTUNE
        # )

        # val_ds = val_ds.map(
        #     lambda x, y: (normalization(x), y),
        #     num_parallel_calls=tensorflow.data.AUTOTUNE
        # )
        

        data_augmentation = tensorflow.keras.Sequential([
            tensorflow.keras.layers.RandomFlip("horizontal"),
            tensorflow.keras.layers.RandomRotation(0.05),
            tensorflow.keras.layers.RandomTranslation(0.1, 0.1),
            tensorflow.keras.layers.RandomZoom(0.1),
            tensorflow.keras.layers.RandomContrast(0.1),
            tensorflow.keras.layers.GaussianNoise(0.02),
        ])

     

        train_batches = tensorflow.data.experimental.cardinality(train_ds).numpy()
        val_batches = tensorflow.data.experimental.cardinality(val_ds).numpy()

        print(f"Train batches per epoch: {train_batches}")
        print(f"Validation batches: {val_batches}")

        print(f"Images per epoch (train): {train_batches * 32}")

        AUTOTUNE = tensorflow.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        
        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tensorflow.data.AUTOTUNE
        )
        
        model = build_classifier_cnn_ball_gopen24_functional()
        model.summary()

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        mlflow.log_param("regularization", True)

        early_stop = tensorflow.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=40,
            restore_best_weights=True
        )

        lr_scheduler = tensorflow.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=15,
            min_lr=1e-6
        )
        
        history = model.fit(train_ds, validation_data=val_ds, epochs=2000, callbacks=[early_stop, lr_scheduler])

        model_path = f"data/{args.camera}/trionda_small_{args.camera}.keras"
        model.save(model_path)

        actual_epochs = history.epoch[-1] + 1
        effective_images = num_train_images * actual_epochs

        print(f"Actual epochs trained: {actual_epochs}")
        print(f"Effective augmented samples seen during training: {effective_images}")
