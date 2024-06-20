import os
import sys
from pathlib import Path

import h5py
import numpy as np

helper_path = os.path.join(os.path.dirname(__file__), "../../")
sys.path.append(helper_path)
import argparse

from tools.helper import (
    ColorMode,
    PatchType,
    combine_datasets_split_train_val,
    download_devils_labeled_patches,
    download_naoth_labeled_patches,
    get_classification_data_devils_bottom,
    get_classification_data_devils_combined,
    get_classification_data_devils_top,
    get_classification_data_naoth_bottom,
    get_classification_data_naoth_combined,
    get_classification_data_naoth_top,
    resize_image_cv2_inter_nearest,
)


def make_data_dir():
    Path(f"{DS_ROOT}").mkdir(parents=True, exist_ok=True)


def download_patches(overwrite=False):
    print("Downloading devils data...")
    download_devils_labeled_patches(save_dir=DEVILS_SAVE_DIR, overwrite=overwrite)

    print("Downloading naoth training data...")
    download_naoth_labeled_patches(
        save_dir=NAOTH_TRAIN_SAVE_DIR,
        validated=True,
        filter_top=naoth_train_top_filter,
        filter_bottom=naoth_train_bottom_filter,
        patch_type=PATCH_TYPE,
        border=BORDER,
        overwrite=overwrite,
    )

    print("Downloading naoth test data...")
    download_naoth_labeled_patches(
        save_dir=NAOTH_TEST_SAVE_DIR,
        validated=False,  # some of the test top projects are not validated yet
        filter_top=naoth_test_top_filter,
        filter_bottom=naoth_test_bottom_filter,
        patch_type=PATCH_TYPE,
        border=BORDER,
        overwrite=overwrite,
    )


def load_and_prepare_data_devils(get_data_func, file_path, color_mode):
    print("Converting data to numpy arrays...")
    X, y = get_data_func(file_path=file_path, color_mode=color_mode)
    y = np.array(y)

    print(f"Resizing images to {PATCH_SIZE}x{PATCH_SIZE}...")
    X = np.array([resize_image_cv2_inter_nearest(img, (PATCH_SIZE, PATCH_SIZE)) for img in X])
    X = X.reshape(-1, PATCH_SIZE, PATCH_SIZE, CHANNELS)

    return X, y


def load_and_prepare_data_naoth(get_data_func, file_path, color_mode, filter_ambiguous_balls=False):
    print("Converting data to numpy arrays...")
    X, y = get_data_func(file_path=file_path, color_mode=color_mode, filter_ambiguous_balls=filter_ambiguous_balls)
    y = np.array(y)

    print(f"Resizing images to {PATCH_SIZE}x{PATCH_SIZE}...")
    X = np.array([resize_image_cv2_inter_nearest(img, (PATCH_SIZE, PATCH_SIZE)) for img in X])
    X = X.reshape(-1, PATCH_SIZE, PATCH_SIZE, CHANNELS)

    return X, y


def save_datasets(DS_BASE_NAME, X_train, y_train, X_val, y_val, X_test, y_test, X_devils, y_devils, X_naoth, y_naoth):
    print("Saving datasets to HDF5 files...")
    with h5py.File(f"{DS_BASE_NAME}_train_ball_no_ball_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_train)
        f.create_dataset("y", data=y_train)

    with h5py.File(f"{DS_BASE_NAME}_validation_ball_no_ball_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_val)
        f.create_dataset("y", data=y_val)

    with h5py.File(f"{DS_BASE_NAME}_test_ball_no_ball_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_test)
        f.create_dataset("y", data=y_test)

    # Save devils data separately
    with h5py.File(f"{DS_BASE_NAME}_devils_ball_no_ball_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_devils)
        f.create_dataset("y", data=y_devils)

    # Save naoth data separately
    with h5py.File(f"{DS_BASE_NAME}_naoth_ball_no_ball_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_naoth)
        f.create_dataset("y", data=y_naoth)


def create_datasets_devils_and_naoth(get_classification_data_devils, get_classification_data_naoth, suffix):
    DS_BASE_NAME = f"{DS_ROOT}/{DS_NAME}_{suffix}_{PATCH_SIZE}x{PATCH_SIZE}"

    X_devils, y_devils = load_and_prepare_data_devils(
        get_classification_data_devils,
        file_path=DEVILS_SAVE_DIR_EXTRACTED,
        color_mode=COLOR_MODE,
    )

    X_naoth, y_naoth = load_and_prepare_data_naoth(
        get_classification_data_naoth,
        file_path=NAOTH_TRAIN_SAVE_DIR,
        color_mode=COLOR_MODE,
        filter_ambiguous_balls=True,
    )

    X_test, y_test = load_and_prepare_data_naoth(
        get_classification_data_naoth,
        file_path=NAOTH_TEST_SAVE_DIR,
        color_mode=COLOR_MODE,
        filter_ambiguous_balls=True,
    )

    print("Merging devils and naoth training data with stratified splitting...")
    X_train, X_val, y_train, y_val = combine_datasets_split_train_val(
        Xs=[X_devils, X_naoth],
        ys=[y_devils, y_naoth],
        test_size=0.15,
        stratify=True,
    )

    print(f"\nX_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Number of ball examples in training set: {y_train.sum()}")

    print(f"\nX_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"Number of ball examples in validation set: {y_val.sum()}")

    print(f"\nX_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Number of ball examples in test set: {y_test.sum()}")

    save_datasets(DS_BASE_NAME, X_train, y_train, X_val, y_val, X_test, y_test, X_devils, y_devils, X_naoth, y_naoth)


def create_datasets_combined():
    create_datasets_devils_and_naoth(
        get_classification_data_devils_combined, get_classification_data_naoth_combined, "combined"
    )


def create_datasets_top():
    create_datasets_devils_and_naoth(get_classification_data_devils_top, get_classification_data_naoth_top, "top")


def create_datasets_bottom():
    create_datasets_devils_and_naoth(
        get_classification_data_devils_bottom, get_classification_data_naoth_bottom, "bottom"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Process some image parameters.")

    parser.add_argument("--patch_size", type=int, default=16, help="Size of the patch")
    parser.add_argument(
        "--patch_type", type=PatchType, choices=list(PatchType), default=PatchType.LEGACY, help="Type of the patch"
    )
    parser.add_argument("--border", type=int, default=0, help="Border size")
    parser.add_argument(
        "--color_mode", type=ColorMode, choices=list(ColorMode), default=ColorMode.YUV422_Y_ONLY_PIL, help="Color mode"
    )
    parser.add_argument("--force-download", action="store_true", default=False, help="Force download of patches")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print("CREATING CLASSIFICATION PATCHES DATASETS")
    print("========================================")
    print(f"PATCH_SIZE = {args.patch_size}")
    print(f"PATCH_TYPE = {args.patch_type}")
    print(f"BORDER = {args.border}")
    print(f"COLOR_MODE = {args.color_mode}")

    PATCH_SIZE = args.patch_size
    PATCH_TYPE = args.patch_type
    BORDER = args.border
    COLOR_MODE = args.color_mode

    CHANNELS = 1 if COLOR_MODE == ColorMode.YUV422_Y_ONLY_PIL else 2

    DS_NAME = f"classification_patches_{COLOR_MODE.value.lower()}_{PATCH_TYPE.name.lower()}_border{BORDER}"
    DS_ROOT = f"../../data/{DS_NAME}/"

    DEVILS_SAVE_DIR = "/tmp/devils_dataset"
    DEVILS_SAVE_DIR_EXTRACTED = f"{DEVILS_SAVE_DIR}/patches_classification_naodevils_32x32x3_GO24"

    NAOTH_SAVE_DIR = f"/tmp/naoth_labeled_patches_ball_center_radius_{PATCH_TYPE.name.lower()}_border{BORDER}"
    NAOTH_TRAIN_SAVE_DIR = f"/tmp/naoth_labeled_patches_train_{PATCH_TYPE.name.lower()}_border{BORDER}"
    NAOTH_TEST_SAVE_DIR = f"/tmp/naoth_labeled_patches_test_{PATCH_TYPE.name.lower()}_border{BORDER}"

    naoth_train_top_filter = "ls_project_top NOT IN ('625', '626', '627', '628')"
    naoth_train_bottom_filter = "ls_project_bottom NOT IN ('629', '630', '631', '632')"
    naoth_test_top_filter = "ls_project_top IN ('625', '626', '627', '628')"
    naoth_test_bottom_filter = "ls_project_bottom IN ('629', '630', '631', '632')"

    make_data_dir()
    download_patches(overwrite=args.force_download)

    print("Creating datasets combined")
    create_datasets_combined()

    print("Creating datasets top")
    create_datasets_top()

    print("Creating datasets bottom")
    create_datasets_bottom()
