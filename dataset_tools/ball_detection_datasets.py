"""
    Tools for creating a dataset suitable to train ball detection networks that can run on the nao

    TODO: I could put all annotations inside the png files. Then i could remove all the csv nonsense
"""
import numpy as np
import h5py
from pathlib import Path

from patch_ball_detection_helper.bhuman import download_bhuman2019, create_classification_dataset, \
    create_detection_dataset
from patch_ball_detection_helper.naoth_patch_helper import create_natural_dataset, create_blender_detection_dataset, \
    create_blender_segmentation_dataset, create_blender_classification_dataset, \
    create_blender_detection_dataset_without_classification, download_tk03_dataset
from patch_ball_detection_helper.common import calculate_mean, subtract_mean, store_output
from common_tools import get_data_root


def generate_bhuman_datasets():
    """
    Downloading the 2019 dataset released by B-Human. Additionally it will create multiple datasets in the format we
    expect for ball training on patches
    """
    # TODO add option to make a balanced dataset
    # TODO add option to only get negative images
    # TODO randomize the result before writing it to a pickle file

    bhuman_root_path = Path(get_data_root()) / "data_balldetection/bhuman"

    # original server is https://sibylle.informatik.uni-bremen.de/public/datasets/b-alls-2019/
    download_bhuman2019("https://logs.naoth.de/Experiments/bhuman/b-alls-2019.hdf5",
                        f"{bhuman_root_path}/b-alls-2019.hdf5")
    download_bhuman2019("https://logs.naoth.de/Experiments/bhuman/readme.txt",
                        f"{bhuman_root_path}/readme.txt")

    # get data
    f = h5py.File(f'{bhuman_root_path}/b-alls-2019.hdf5', 'r')

    negative_data = np.array(f.get('negatives/data'))
    positive_data = np.array(f.get('positives/data'))
    negative_labels = np.array(f.get('negatives/labels'))
    positive_labels = np.array(f.get('positives/labels'))

    create_classification_dataset(bhuman_root_path, negative_data, positive_data)
    create_detection_dataset(bhuman_root_path, negative_data, positive_data, negative_labels, positive_labels)


def create_tk03_classification_datasets():
    """
    Creates 3 different versions of the tk03 dataset with classification labels.
    - only recorded patches
    - only blender created patches
    - combined patches
    """
    naoth_root_path = Path(get_data_root()) / "data_balldetection/naoth"
    tk03_path = Path(get_data_root()) / "data_balldetection/naoth/TK-03"
    print(tk03_path)
    if not Path(tk03_path).exists():
        download_tk03_dataset()

    # TODO: what is the res really needed for here?
    # TODO think of improvements for the balancing
    res = {"x": 16, "y": 16}
    limit_noball = True

    x, y, p = create_natural_dataset(tk03_path, res, limit_noball, "classification")
    mean = calculate_mean(x)
    x_mean = subtract_mean(x, mean)

    print("save classification dataset with natural images")
    output_name = str(naoth_root_path / 'tk03_natural_classification.pkl')
    store_output(output_name, mean, x_mean, y, p)

    path = Path(tk03_path) / "blender"

    x_syn, y_syn = create_blender_classification_dataset(str(path), res)
    mean_b = calculate_mean(x_syn)
    x_syn_mean = subtract_mean(x_syn, mean_b)

    print("save classification dataset with synthetic images")
    output_name = str(naoth_root_path / 'tk03_synthetic_classification.pkl')
    store_output(output_name, mean_b, x_syn_mean, y_syn)  # FIXME paths are missing here

    # merge the two datasets
    X = np.concatenate((x, x_syn))
    Y = np.concatenate((y, y_syn))
    combined_mean = calculate_mean(X)
    X_mean = X - combined_mean

    print("save classification dataset with combined images")
    output_name = str(naoth_root_path / 'tk03_combined_classification.pkl')
    store_output(output_name, combined_mean, X_mean, Y)  # FIXME paths are missing here
    #----------------------------------------------
    # Do simple balancing here
    print("len x", len(x))
    print("len x_syn", len(x_syn))
    x_new_syn = x_syn[:len(x)]
    y_new_syn = y_syn[:len(x)]
    print(len(x_new_syn))

    # merge the original and balanced synthetic datasets
    X_NEW = np.concatenate((x, x_new_syn))
    Y_NEW = np.concatenate((y, y_new_syn))
    combined_mean_new = calculate_mean(X_NEW)
    X_NEW_mean = X_NEW - combined_mean

    print("save classification dataset with balanced combined images")
    output_name = str(naoth_root_path / 'tk03_combined_balanced_classification.pkl')
    store_output(output_name, combined_mean_new, X_NEW_mean, Y_NEW)  # FIXME paths are missing here
    #----------------------------------------------


def create_tk03_detection_datasets():
    """
    Creates 3 different versions of the tk03 dataset with detection labels.
    - only recorded patches
    - only blender created patches
    - combined patches

    The target vector y has the following format:
    radius, x_coord, y_coord, is_ball

    is_ball is either 1.0 or 0.0
    """
    naoth_root_path = Path(get_data_root()) / "data_balldetection/naoth"
    tk03_path = Path(get_data_root()) / "data_balldetection/naoth/TK-03"

    if not Path(naoth_root_path / "TK-03").exists():
        download_tk03_dataset()

    # TODO: what is the res really needed for here?
    # TODO think of improvements for the balancing
    res = {"x": 16, "y": 16}
    limit_noball = True

    x, y, p = create_natural_dataset(tk03_path, res, limit_noball, "detection")
    print("sum:", np.sum(y))
    mean = calculate_mean(x)
    x_mean = subtract_mean(x, mean)

    print("save detection dataset with natural images")
    output_name = str(naoth_root_path / 'tk03_natural_detection.pkl')
    store_output(output_name, mean, x_mean, y, p)

    path = Path(tk03_path) / "blender"
    x_syn, y_syn, p_syn = create_blender_detection_dataset(str(path), res)
    mean_b = calculate_mean(x_syn)
    x_syn_mean = subtract_mean(x_syn, mean_b)

    print("save detection dataset with synthetic images")
    output_name = str(naoth_root_path / 'tk03_synthetic_detection.pkl')
    store_output(output_name, mean_b, x_syn_mean, y_syn, p_syn)

    # merge the two datasets
    X = np.concatenate((x, x_syn))
    Y = np.concatenate((y, y_syn))
    P = np.concatenate((p, p_syn))
    combined_mean = calculate_mean(X)
    X_mean = X - combined_mean

    print("save detection dataset with combined images")
    output_name = str(naoth_root_path / 'tk03_combined_detection.pkl')
    store_output(output_name, combined_mean, X_mean, Y, P)


def create_tk03_detection2_datasets():
    """
    Creates 3 different versions of the tk03 dataset with detection labels.
    - only recorded patches
    - only blender created patches
    - combined patches

    The target vector y has the following format:
    radius, x_coord, y_coord

    NOTE: here the classification is missing. This is useful if you filter out the non ball patches with another
    classification network before.
    """
    naoth_root_path = Path(get_data_root()) / "data_balldetection/naoth"
    tk03_path = Path(get_data_root()) / "data_balldetection/naoth/TK-03"

    if not Path(naoth_root_path / "TK-03").exists():
        download_tk03_dataset()

    # TODO: what is the res really needed for here?
    # TODO think of improvements for the balancing
    res = {"x": 16, "y": 16}
    limit_noball = True

    x, y, p = create_natural_dataset(tk03_path, res, limit_noball, "detection2")
    print("sum:", np.sum(y))
    mean = calculate_mean(x)
    x_mean = subtract_mean(x, mean)

    print("save detection dataset with natural images")
    output_name = str(naoth_root_path / 'tk03_natural_detection2.pkl')
    store_output(output_name, mean, x_mean, y, p)

    path = Path(tk03_path) / "blender"
    x_syn, y_syn, p_syn = create_blender_detection_dataset_without_classification(str(path), res)
    mean_b = calculate_mean(x_syn)
    x_syn_mean = subtract_mean(x_syn, mean_b)

    print("save detection dataset with synthetic images")
    output_name = str(naoth_root_path / 'tk03_synthetic_detection2.pkl')
    store_output(output_name, mean_b, x_syn_mean, y_syn, p_syn)

    # merge the two datasets
    X = np.concatenate((x, x_syn))
    Y = np.concatenate((y, y_syn))
    P = np.concatenate((p, p_syn))
    combined_mean = calculate_mean(X)
    X_mean = X - combined_mean

    print("save detection dataset with combined images")
    output_name = str(naoth_root_path / 'tk03_combined_detection2.pkl')
    store_output(output_name, combined_mean, X_mean, Y, P)


def create_tk03_segmentation_datasets():
    """
    Creates 3 different versions of the tk03 dataset with segmentation labels.
    - only recorded patches
    - only blender created patches
    - combined patches

    The target y has the same format as the input image
    """
    naoth_root_path = Path(get_data_root()) / "data_balldetection/naoth"
    tk03_path = Path(get_data_root()) / "data_balldetection/naoth/TK-03"

    if not Path(naoth_root_path / "TK-03").exists():
        download_tk03_dataset()

    # TODO: what is the res really needed for here?
    # TODO think of improvements for the balancing
    res = {"x": 16, "y": 16}
    limit_noball = True

    x, y, p = create_natural_dataset(tk03_path, res, limit_noball, "segmentation")
    mean = calculate_mean(x)
    x_mean = subtract_mean(x, mean)

    print("save segmentation dataset with natural images")
    output_name = str(naoth_root_path / 'tk03_natural_segmentation.pkl')
    store_output(output_name, mean, x_mean, y, p)

    path = Path(tk03_path) / "blender"
    x_syn, y_syn = create_blender_segmentation_dataset(str(path), res)
    mean_b = calculate_mean(x_syn)
    x_syn_mean = subtract_mean(x_syn, mean_b)

    output_name = str(naoth_root_path / 'tk03_synthetic_segmentation.pkl')
    store_output(output_name, mean_b, x_syn_mean, y_syn)

    # merge the two datasets
    X = np.concatenate((x, x_syn))
    Y = np.concatenate((y, y_syn))
    combined_mean = calculate_mean(X)
    X_mean = X - combined_mean

    print("save detection dataset with combined images")
    output_name = str(naoth_root_path / 'tk03_combined_segmentation.pkl')
    store_output(output_name, combined_mean, X_mean, Y)

create_tk03_classification_datasets()

#create_tk03_segmentation_datasets()
#create_tk03_detection2_datasets()
#create_tk03_detection_datasets()
