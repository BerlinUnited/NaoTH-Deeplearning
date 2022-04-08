"""
    The functions create datasets for the new cppyy patch data

    TODO implement resizing here
    TODO implement the new simpler folder structure
"""
from pathlib import Path
import cv2
import numpy as np
import pickle

from common_tools import get_data_root


def create_2019_patch_dataset(patch_size=16, color=False, iou_tresh = 0.3):
    """
    """
    naoth_root_path = Path(get_data_root()) / "data_balldetection/naoth"
    rc19_path = Path(get_data_root()) / "data_cvat/RoboCup2019/combined/COCO_1.0/"

    ball_images = list()
    ball_targets = list()

    noball_images = list()
    noball_targets = list()

    color_string = "color" if color else "bw"

    # get all dataset paths
    subfu = [f for f in rc19_path.iterdir() if f.is_dir()]
    for folder in subfu:
        patch_paths = Path(folder / "all_patches").glob('**/*.png')
        for patch_path in patch_paths:
            if color:
                img_cv = cv2.imread(str(patch_path))
            else:
                img_cv = cv2.imread(str(patch_path), cv2.IMREAD_GRAYSCALE)
            img_normalized = img_cv.astype(float) / 255.0

            # TODO load meta data

            # TODO resize here

            # TODO change this according to iou and iou_thresh
            target = np.array([1.0]) # ball
            ball_images.append(img_normalized)
            ball_targets.append(target)

    # TODO balancing here
    # TODO add option for per image mean vs global mean vs no mean
    all_images = ball_images + noball_images
    mean_all_images = np.mean(all_images)
    all_images_wo_mean = all_images - mean_all_images
    # expand dimensions of the input images for use with tensorflow
    all_images_wo_mean = all_images_wo_mean.reshape(*all_images_wo_mean.shape, 1)

    all_targets = np.array(ball_targets + noball_targets)

    output_name = str(naoth_root_path / f'rc19_classification_{patch_size}_{color_string}.pkl')
    with open(output_name, "wb") as f:
        pickle.dump(mean_all_images, f)
        pickle.dump(all_images_wo_mean, f)
        pickle.dump(all_targets, f)


create_2019_patch_dataset(8, color=True)
create_2019_patch_dataset(8, color=False)
create_2019_patch_dataset(16, color=True)
create_2019_patch_dataset(16, color=False)
create_2019_patch_dataset(32, color=True)
create_2019_patch_dataset(32, color=False)
create_2019_patch_dataset(64, color=True)
create_2019_patch_dataset(64, color=False)