import pickle
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import load_model

from nncg.nncg import NNCG

tf.compat.v1.disable_eager_execution()  # TODO why was this necessary???


def main():

    images = {}
    data_file = "C://RoboCup//Datasets//data_balldetection//naoth//tk03_natural_detection.pkl"
    with open(data_file, "rb") as f:
        images["mean"] = pickle.load(f)
        images["images"] = pickle.load(f)
        images["y"] = pickle.load(f)

    model = load_model("dummy_model2.h5")
    nncg = NNCG()
    nncg.keras_compile(imdb=images['images'], model=model, code_path="dummy_model2.cpp", arch="sse3")


if __name__ == '__main__':
    main()
