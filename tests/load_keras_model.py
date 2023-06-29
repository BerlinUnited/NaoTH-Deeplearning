"""

"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
import time
import tensorflow.keras as keras

def f1():
    pass

if __name__ == '__main__':
    model = keras.models.load_model("players_deeptector.h5", custom_objects={"f2": f1})
    model.summary()