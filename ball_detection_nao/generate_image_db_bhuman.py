"""
Converts the b-human 2019 dataset to the naoth format so we can run performance comparisons

TODO add option to make a balanced dataset
TODO add option to only get negative images
"""
import pickle
import numpy as np
import h5py
import toml
from pathlib import Path

from utility_functions.loader import calculate_mean, subtract_mean
from utility_functions.bhuman_helper import download_bhuman2019


def create_classification_dataset(data_root_path, negative_data, positive_data):
    images = np.append(negative_data, positive_data, axis=0)
    mean = calculate_mean(images)
    mean_images = subtract_mean(images, mean)

    negative_labels = [0] * len(negative_data)
    positive_labels = [1] * len(positive_data)
    labels = negative_labels + positive_labels

    with open(f"{data_root_path}/bhuman_classification.pkl", "wb") as f:
        pickle.dump(mean, f)
        pickle.dump(mean_images, f)
        pickle.dump(np.array(labels), f)


def create_detection_dataset(data_root_path, negative_data, positive_data, negative_labels, positive_labels):
    labels = np.append(negative_labels, positive_labels, axis=0)

    # swap dimensions to convert b-human format to berlin-united format
    new_labels = np.copy(labels)
    radii = labels[:, 0]
    classes = labels[:, -1]
    new_labels[:, 0] = classes
    new_labels[:, -1] = radii

    images = np.append(negative_data, positive_data, axis=0)
    mean = calculate_mean(images)

    mean_images = subtract_mean(images, mean)
    with open(f"{data_root_path}/bhuman_detection.pkl", "wb") as f:
        pickle.dump(mean, f)
        pickle.dump(mean_images, f)
        pickle.dump(new_labels, f)


if __name__ == '__main__':
    # load data folder from config file
    with open('classification.toml', 'r') as f:
        config_dict = toml.load(f)
    cfg = config_dict["classification_1"]

    data_root_path = Path(cfg["data_root_path"]).resolve()
 
    # original server is https://sibylle.informatik.uni-bremen.de/public/datasets/b-alls-2019/
    download_bhuman2019("https://logs.naoth.de/Experiments/bhuman/b-alls-2019.hdf5",
                        f"{data_root_path}/bhuman/b-alls-2019.hdf5")
    download_bhuman2019("https://logs.naoth.de/Experiments/bhuman/readme.txt",
                        f"{data_root_path}/bhuman/readme.txt")

    # get data
    f = h5py.File(f'{data_root_path}/bhuman/b-alls-2019.hdf5', 'r')
    
    negative_data = np.array(f.get('negatives/data'))
    positive_data = np.array(f.get('positives/data'))
    negative_labels = np.array(f.get('negatives/labels'))
    positive_labels = np.array(f.get('positives/labels'))
    
    create_detection_dataset(data_root_path, negative_data, positive_data, negative_labels, positive_labels)
    create_classification_dataset(data_root_path, negative_data, positive_data)