"""
    The functions create datasets for the new cppyy patch data
"""

from pathlib import Path
import cv2
import numpy as np
import pickle
from PIL import Image

from common_tools.main import get_data_root


def create_2019_patch_dataset(patch_size=16, color=False, iou_tresh=0.3, cam_combined=True):
    """ """
    naoth_root_path = Path(get_data_root()) / "data_balldetection/naoth"
    rc19_path = Path(get_data_root()) / "data_cvat/RoboCup2019/combined/COCO_1.0/"
    color_string = "color" if color else "bw"

    top_ball_images = list()
    top_ball_targets = list()
    bottom_ball_images = list()
    bottom_ball_targets = list()

    top_noball_images = list()
    top_noball_targets = list()
    bottom_noball_images = list()
    bottom_noball_targets = list()

    # get all dataset paths
    # TODO the for loops can go into a function
    subfu = [f for f in rc19_path.iterdir() if f.is_dir()]
    for folder in subfu:
        patch_paths = Path(folder / "all_patches").glob("**/*.png")
        for patch_path in patch_paths:
            if color:
                img_cv = cv2.imread(str(patch_path))
            else:
                img_cv = cv2.imread(str(patch_path), cv2.IMREAD_GRAYSCALE)
            img_normalized = img_cv.astype(float) / 255.0

            # load meta data from image header
            img_pil = Image.open(str(patch_path))
            bottom = img_pil.info["CameraID"] == "1"
            iou = float(img_pil.info["iou"])
            center_x = float(img_pil.info["center_x"])
            center_y = float(img_pil.info["center_y"])
            radius = float(img_pil.info["radius"])

            # We need to resize always since the patches are in their original form
            x_ratio = patch_size / img_cv.shape[1]
            y_ratio = patch_size / img_cv.shape[0]
            center_x = round(center_x * x_ratio)
            center_y = round(center_y * y_ratio)
            radius = round(radius * x_ratio)  # TODO this must be handled differently if ratios are not the same

            img_cv_resized = cv2.resize(img_normalized, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
            # TODO build a visualisation here

            # TODO make it possible to build detection datasets here
            if iou >= iou_tresh:
                target = np.array([1.0])  # ball
                if bottom:
                    bottom_ball_images.append(img_cv_resized)
                    bottom_ball_targets.append(target)
                else:
                    top_ball_images.append(img_cv_resized)
                    top_ball_targets.append(target)
            else:
                target = np.array([0.0])  # ball
                if bottom:
                    bottom_noball_images.append(img_cv_resized)
                    bottom_noball_targets.append(target)
                else:
                    top_noball_images.append(img_cv_resized)
                    top_noball_targets.append(target)

    """
    # create validation datasets. For better evaluation i split the bottom and top images
    """
    # create bottom eval dataset
    val_img, val_target = create_val_sets(
        bottom_ball_images, bottom_noball_images, bottom_ball_targets, bottom_noball_targets
    )
    output_name = str(naoth_root_path / f"rc19_classification_{patch_size}_{color_string}_val_btm.pkl")
    save_datasets(val_img, val_target, output_name)

    # create top eval dataset
    val_img, val_target = create_val_sets(top_ball_images, top_noball_images, top_ball_targets, top_noball_targets)
    output_name = str(naoth_root_path / f"rc19_classification_{patch_size}_{color_string}_val_top.pkl")
    save_datasets(val_img, val_target, output_name)

    """
    TODO balancing here
    # boilerplate code for balancing
    btm_img, btm_targets = balance(bottom_ball_images, bottom_noball_images, bottom_ball_targets, bottom_noball_targets)
    top_img, top_targets = balance(top_ball_images, top_noball_images, top_ball_targets, top_noball_targets)
    
    if cam_combined:
        all_img, all_targets = balance(btm_img, btm_targets, top_img, top_targets)
    """
    all_bottom_images = bottom_ball_images + bottom_noball_images
    all_top_images = top_ball_images + top_noball_images
    all_bottom_targets = bottom_ball_targets + bottom_noball_targets
    all_top_targets = top_ball_targets + top_noball_targets

    if cam_combined:
        all_images = all_bottom_images + all_top_images
        all_targets = all_bottom_targets + all_top_targets

        output_name = str(naoth_root_path / f"rc19_classification_{patch_size}_{color_string}_combined.pkl")
        save_datasets(all_images, all_targets, output_name)
    else:
        output_name = str(naoth_root_path / f"rc19_classification_{patch_size}_{color_string}_bottom.pkl")
        save_datasets(all_bottom_images, all_bottom_targets, output_name)
        output_name = str(naoth_root_path / f"rc19_classification_{patch_size}_{color_string}_top.pkl")
        save_datasets(all_top_images, all_top_targets, output_name)


def create_val_sets(ball_img, noball_img, ball_targets, noball_targets):
    # TODO maybe something more random should go here
    val_img = ball_img[0:100] + noball_img[0:100]
    val_targets = ball_targets[0:100] + noball_targets[0:100]

    return val_img, val_targets


def balance_datases(ball_img, noball_img, ball_targets, noball_targets):
    pass


def handle_mean(images, mean_flag="global"):
    """
    TODO describe what the format of images should be
    Note that the devils compiler do make assumptions about mean stuff. See predict function in the generated cpp file
    """
    mean_val = np.mean(images)
    all_images_wo_mean = images - mean_val
    return all_images_wo_mean, mean_val


def save_datasets(all_images, targets, output_name):
    # TODO add option for per image mean vs global mean vs no mean. Need to handle the pickle format correctly in training
    all_images = np.array(all_images)
    all_images, mean_val = handle_mean(all_images)

    # expand dimensions of the input images for use with tensorflow
    all_images = all_images.reshape(*all_images.shape, 1)
    all_targets = np.array(targets)

    # shuffle data
    # TODO set random seed
    p = np.random.permutation(len(all_images))
    all_images = all_images[p]
    all_targets = all_targets[p]

    with open(output_name, "wb") as f:
        pickle.dump(mean_val, f)
        pickle.dump(all_images, f)
        pickle.dump(all_targets, f)


if __name__ == "__main__":
    # Top vs bottom vs combined experiment
    # create_2019_patch_dataset(16, color=False, cam_combined=False)
    # create_2019_patch_dataset(16, color=False, cam_combined=True)

    # color experiment
    # create_2019_patch_dataset(16, color=True, cam_combined=True)

    # patch size experiment
    # create_2019_patch_dataset(8, color=False, cam_combined=True)
    # create_2019_patch_dataset(12, color=False, cam_combined=True)
    # create_2019_patch_dataset(24, color=False, cam_combined=True)
    # create_2019_patch_dataset(32, color=False, cam_combined=True)
    pass
