import shutil
from pathlib import Path
import sys
import argparse
import tensorflow as tf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", type=str, help="Set BOTTOM or TOP")
    args = parser.parse_args()

    if args.camera is None:
        print("The camera is not set.\nSet with option -c, --camera TOP/BOTTOM")
        sys.exit()

    model_path = f"data/{args.camera}/trionda_small_{args.camera}.keras"

    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    output_file = Path(model_path).with_suffix(".tflite")
    with open(output_file, "wb") as f:
        f.write(tflite_model)
