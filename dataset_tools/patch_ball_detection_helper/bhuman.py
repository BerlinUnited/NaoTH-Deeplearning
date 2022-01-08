import numpy as np
import pickle
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError
import cv2

from .common import calculate_mean, subtract_mean


# TODO rename and unify this with other locations where stuff is downloaded directly
def download_bhuman2019(origin, target):
    def dl_progress(count, block_size, total_size):
        print('\r', 'Progress: {0:.2%}'.format(min((count * block_size) / total_size, 1.0)), sep='', end='', flush=True)

    if not Path(target).exists():
        target_folder = Path(target).parent
        target_folder.mkdir(parents=True, exist_ok=True)
    else:
        return

    error_msg = 'URL fetch failure on {} : {} -- {}'
    try:
        try:
            urlretrieve(origin, target, dl_progress)
            print('\nFinished')
        except HTTPError as e:
            raise Exception(error_msg.format(origin, e.code, e.reason))
        except URLError as e:
            raise Exception(error_msg.format(origin, e.errno, e.reason))
    except (Exception, KeyboardInterrupt):
        if Path(target).exists():
            Path(target).unlink()
        raise


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
