import numpy as np
import pickle
from pathlib import Path
import cv2
import h5py

from core.main import get_data_root, download_function


def calculate_mean(images):
    return np.mean(images)


def subtract_mean(images, mean):
    return images - mean


def create_classification_dataset(data_root_path, negative_data, positive_data):
    images = np.append(positive_data, negative_data, axis=0)
    images = images / 255.0
    """
    min_val = list()
    max_val = list()
    for image in images:
        min_val.append(image.min())
        max_val.append(image.max())
    print(np.array(min_val).min())
    print(np.array(max_val).max())
    """

    negative_labels = [0] * len(negative_data)
    positive_labels = [1] * len(positive_data)
    labels = np.array(positive_labels + negative_labels)

    # initialize own pseudorandom number generator with seed
    rng = np.random.default_rng(42)
    indices = np.arange(images.shape[0])
    rng.shuffle(indices)

    images = images[indices]
    labels = labels[indices]

    mean = calculate_mean(images)
    mean_images = subtract_mean(images, mean)

    with open(f"{data_root_path}/bhuman_classification_32x32.pkl", "wb") as f:
        pickle.dump(mean, f)
        pickle.dump(mean_images, f)
        pickle.dump(labels, f)

    # resize bhuman images from 32x32 to 16x16
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
        pickle.dump(labels, f)


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
    with open(f"{data_root_path}/bhuman_detection_32x32.pkl", "wb") as f:
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


def generate_bhuman_ball_datasets_2019():
    """
    Downloading the 2019 dataset released by B-Human. Additionally it will create multiple datasets in the format we
    expect for ball training on patches
    """
    # TODO add option to make a balanced dataset
    # TODO add option to only get negative images
    # TODO randomize the result before writing it to a pickle file

    bhuman_root_path = Path(get_data_root()) / "data_balldetection/bhuman"

    # original server is https://sibylle.informatik.uni-bremen.de/public/datasets/b-alls-2019/
    download_function(
        "https://logs.naoth.de/Experiments/bhuman/b-alls-2019.hdf5", f"{bhuman_root_path}/b-alls-2019.hdf5"
    )
    download_function("https://logs.naoth.de/Experiments/bhuman/readme.txt", f"{bhuman_root_path}/readme.txt")

    # get data
    f = h5py.File(f"{bhuman_root_path}/b-alls-2019.hdf5", "r")

    negative_data = np.array(f.get("negatives/data"))
    positive_data = np.array(f.get("positives/data"))
    negative_labels = np.array(f.get("negatives/labels"))
    positive_labels = np.array(f.get("positives/labels"))

    create_classification_dataset(bhuman_root_path, negative_data, positive_data)
    create_detection_dataset(bhuman_root_path, negative_data, positive_data, negative_labels, positive_labels)


if __name__ == "__main__":
    generate_bhuman_ball_datasets_2019()
