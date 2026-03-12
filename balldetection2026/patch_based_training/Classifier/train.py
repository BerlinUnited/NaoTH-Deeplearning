import argparse
import json
import os
import sys
import re
import sys
from pathlib import Path

import mlflow
import tensorflow

from balldetection2026.patch_based_training.model import build_classifier_cnn_ball_gopen24_functional

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

        val_image_names = set()
        for path in val_ds.file_paths:
            filename = Path(path).stem

            base_stem = re.sub(r"_(no)?ball_\d+$", "", filename)
            val_image_names.add(f"{base_stem}.png")

        list_path = f"data/{args.camera}/val_images.json"
        with open(list_path, "w") as f:
            json.dump(list(val_image_names), f, indent=4)

        model = build_classifier_cnn_ball_gopen24_functional()
        model.summary()

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        mlflow.log_param("regularization", True)

        model.fit(train_ds, validation_data=val_ds, epochs=2000)

        model_path = f"data/{args.camera}/trionda_small_{args.camera}.keras"
        model.save(model_path)
