"""
Converts the b-human 2019 dataset to the naoth format so we can run performance comparisons

TODO add option to make a balanced dataset
TODO add option to only get negative images
TODO randomize the result before writing it to a pickle file
"""
import pickle
import numpy as np
import h5py
import toml
from pathlib import Path
import cv2
import random

from utility_functions.loader import calculate_mean, subtract_mean
from utility_functions.bhuman_helper import download_bhuman2019


def create_classification_dataset(data_root_path, negative_data, positive_data):
    images = np.append(positive_data,negative_data,  axis=0)
    mean = calculate_mean(images)
    mean_images = subtract_mean(images, mean)

    negative_labels = [0] * len(negative_data)
    positive_labels = [1] * len(positive_data)
    labels = positive_labels + negative_labels
    
    # randomize the images and the labels
    c = list(zip(mean_images, labels))
    random.shuffle(c)
    mean_images, labels = zip(*c)

    with open(f"{data_root_path}/bhuman_classification.pkl", "wb") as f:
        pickle.dump(mean, f)
        pickle.dump(mean_images, f)
        pickle.dump(np.array(labels), f)
    
    # resize bhuman images from 32x32 to 16x16

    # FIXME randomize is not used here
    smaller_data = list()
    for image in images:
        new_image = cv2.resize(image, dsize=(16, 16), interpolation=cv2.INTER_CUBIC)
        new_image = new_image.reshape(*new_image.shape, 1)

        smaller_data.append(new_image)

    smaller_data = np.array(smaller_data)
    new_mean = calculate_mean(smaller_data)
    new_mean_images = subtract_mean(smaller_data, new_mean)

    with open(f"{data_root_path}/bhuman_classification_16x16.pkl", "wb") as f:
        pickle.dump(new_mean, f)
        pickle.dump(new_mean_images, f)
        pickle.dump(np.array(labels), f)


def create_detection_dataset(data_root_path, negative_data, positive_data, negative_labels, positive_labels):
    # FIXME randomize order with fixed seed
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

    # resize bhuman images from 32x32 to 16x16
    smaller_data = list()
    for image in images:
        new_image = cv2.resize(image, dsize=(16, 16), interpolation=cv2.INTER_CUBIC)
        new_image = new_image.reshape(*new_image.shape, 1)

        smaller_data.append(new_image)

    smaller_data = np.array(smaller_data)
    new_mean = calculate_mean(smaller_data)
    new_mean_images = subtract_mean(smaller_data, new_mean)

    # FIXME resize labels
    with open(f"{data_root_path}/bhuman_detection_16x16.pkl", "wb") as f:
        pickle.dump(new_mean, f)
        pickle.dump(new_mean_images, f)
        pickle.dump(new_labels, f)


if __name__ == '__main__':
    # load data folder from config file
    with open('classification.toml', 'r') as f:
        config_dict = toml.load(f)
    cfg = config_dict["classification_2"]

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
    
    create_classification_dataset(data_root_path, negative_data, positive_data)
    #create_detection_dataset(data_root_path, negative_data, positive_data, negative_labels, positive_labels)
