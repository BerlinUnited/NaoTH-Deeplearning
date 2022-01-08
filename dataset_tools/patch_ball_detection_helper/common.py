import numpy as np
import pickle


def calculate_mean(images):
    return np.mean(images)


def subtract_mean(images, mean):
    return images - mean


def store_output(output_file, mean, x, y, p=None):
    with open(output_file, "wb") as f:
        pickle.dump(mean, f)
        pickle.dump(x, f)
        pickle.dump(y, f)
        pickle.dump(p, f)
