import os
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

helper_path = os.path.join(os.path.dirname(__file__), "../../")
sys.path.append(helper_path)

import argparse

from tools.helper import (
    ColorMode,
    PatchType,
    combine_datasets_split_train_val,
    download_naoth_labeled_patches,
    download_tk_03_dataset,
    get_ball_center_radius_data_naoth_bottom,
    get_ball_center_radius_data_naoth_combined,
    get_ball_center_radius_data_naoth_top,
    get_ball_center_radius_data_tk_03_combined,
    resize_image_cv2_inter_nearest,
)


def make_data_dir():
    Path(f"{DS_ROOT}").mkdir(parents=True, exist_ok=True)


def download_tk03_patches(overwrite=False):
    print("Downloading tk03 detection training data...")
    download_tk_03_dataset(
        save_dir=TK_SAVE_DIR, url="https://datasets.naoth.de/tk03_combined_detection.pkl", overwrite=overwrite
    )


def download_patches(overwrite=False):
    print("Downloading naoth detection training data...")
    download_naoth_labeled_patches(
        save_dir=NAOTH_SAVE_DIR, validated=True, patch_type=PATCH_TYPE, border=BORDER, overwrite=overwrite
    )


def load_and_prepare_data_naoth(get_data_func, file_path, color_mode, filter_ambiguous_balls=False):
    print("Converting data to numpy arrays...")
    X, y = get_data_func(file_path=file_path, color_mode=color_mode, filter_ambiguous_balls=filter_ambiguous_balls)
    y = np.array(y)

    print(f"Resizing images to {PATCH_SIZE}x{PATCH_SIZE}...")
    X = np.array([resize_image_cv2_inter_nearest(img, (PATCH_SIZE, PATCH_SIZE)) for img in X])
    X = X.reshape(-1, PATCH_SIZE, PATCH_SIZE, CHANNELS)

    return X, y


def create_datasets_tk03_and_naoth_combined():
    DS_BASE_NAME = f"{DS_ROOT}/{DS_NAME}_combined_{PATCH_SIZE}x{PATCH_SIZE}"

    X_naoth, y_naoth = load_and_prepare_data_naoth(
        get_ball_center_radius_data_naoth_combined,
        file_path=NAOTH_SAVE_DIR,
        color_mode=COLOR_MODE,
        filter_ambiguous_balls=True,
    )

    X_tk, y_tk = get_ball_center_radius_data_tk_03_combined(file_path=TK_FILE_PATH, balls_only=True)
    X_tk = np.array([resize_image_cv2_inter_nearest(img, (PATCH_SIZE, PATCH_SIZE)) for img in X_tk])
    X_tk = X_tk.reshape(-1, PATCH_SIZE, PATCH_SIZE, CHANNELS)
    y_tk = np.array(y_tk)

    print(f"\nX_naoth shape: {X_naoth.shape}")
    print(f"y_naoth shape: {y_naoth.shape}")

    print(f"\nX_tk shape: {X_tk.shape}")
    print(f"y_tk shape: {y_tk.shape}")

    X_train, X_val, y_train, y_val = combine_datasets_split_train_val(
        Xs=[X_tk, X_naoth],
        ys=[y_tk, y_naoth],
        test_size=0.15,
        stratify=False,
    )

    print(f"\nX_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    print(f"\nX_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")

    with h5py.File(f"{DS_BASE_NAME}_tk03_ball_center_radius_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_naoth)
        f.create_dataset("y", data=y_naoth)

    with h5py.File(f"{DS_BASE_NAME}_naoth_ball_center_radius_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_naoth)
        f.create_dataset("y", data=y_naoth)

    with h5py.File(f"{DS_BASE_NAME}_train_ball_center_radius_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_train)
        f.create_dataset("y", data=y_train)

    with h5py.File(f"{DS_BASE_NAME}_validation_ball_center_radius_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_val)
        f.create_dataset("y", data=y_val)


def create_datasets_combined():
    DS_BASE_NAME = f"{DS_ROOT}/{DS_NAME}_combined_{PATCH_SIZE}x{PATCH_SIZE}"

    X_naoth, y_naoth = load_and_prepare_data_naoth(
        get_ball_center_radius_data_naoth_combined,
        file_path=NAOTH_SAVE_DIR,
        color_mode=COLOR_MODE,
        filter_ambiguous_balls=True,
    )

    print(f"\nX_naoth shape: {X_naoth.shape}")
    print(f"y_naoth shape: {y_naoth.shape}")

    X_train, X_val, y_train, y_val = train_test_split(X_naoth, y_naoth, test_size=0.15, shuffle=True, random_state=42)

    print(f"\nX_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    print(f"\nX_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")

    with h5py.File(f"{DS_BASE_NAME}_naoth_ball_center_radius_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_naoth)
        f.create_dataset("y", data=y_naoth)

    with h5py.File(f"{DS_BASE_NAME}_train_ball_center_radius_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_train)
        f.create_dataset("y", data=y_train)

    with h5py.File(f"{DS_BASE_NAME}_validation_ball_center_radius_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_val)
        f.create_dataset("y", data=y_val)


def create_datasets_top():
    DS_BASE_NAME = f"{DS_ROOT}/{DS_NAME}_top_{PATCH_SIZE}x{PATCH_SIZE}"

    X_naoth, y_naoth = load_and_prepare_data_naoth(
        get_ball_center_radius_data_naoth_top,
        file_path=NAOTH_SAVE_DIR,
        color_mode=COLOR_MODE,
        filter_ambiguous_balls=True,
    )
    print(f"\nX_naoth shape: {X_naoth.shape}")
    print(f"y_naoth shape: {y_naoth.shape}")

    X_train, X_val, y_train, y_val = train_test_split(X_naoth, y_naoth, test_size=0.15, shuffle=True, random_state=42)

    print(f"\nX_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    print(f"\nX_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")

    with h5py.File(f"{DS_BASE_NAME}_naoth_ball_center_radius_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_naoth)
        f.create_dataset("y", data=y_naoth)

    with h5py.File(f"{DS_BASE_NAME}_train_ball_center_radius_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_train)
        f.create_dataset("y", data=y_train)

    with h5py.File(f"{DS_BASE_NAME}_validation_ball_center_radius_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_val)
        f.create_dataset("y", data=y_val)


def create_datasets_bottom():
    DS_BASE_NAME = f"{DS_ROOT}/{DS_NAME}_bottom_{PATCH_SIZE}x{PATCH_SIZE}"

    X_naoth, y_naoth = load_and_prepare_data_naoth(
        get_ball_center_radius_data_naoth_bottom,
        file_path=NAOTH_SAVE_DIR,
        color_mode=COLOR_MODE,
        filter_ambiguous_balls=True,
    )

    X_train, X_val, y_train, y_val = train_test_split(X_naoth, y_naoth, test_size=0.15, shuffle=True, random_state=42)

    print(f"\nX_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    print(f"\nX_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")

    print(f"\nX_naoth shape: {X_naoth.shape}")
    print(f"y_naoth shape: {y_naoth.shape}")

    with h5py.File(f"{DS_BASE_NAME}_naoth_ball_center_radius_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_naoth)
        f.create_dataset("y", data=y_naoth)

    with h5py.File(f"{DS_BASE_NAME}_train_ball_center_radius_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_train)
        f.create_dataset("y", data=y_train)

    with h5py.File(f"{DS_BASE_NAME}_validation_ball_center_radius_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_val)
        f.create_dataset("y", data=y_val)


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
    parser.add_argument("--force_download", action="store_true", default=False, help="Force download of patches")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print("CREATING BALL CENTER RADIUS PATCHES DATASETS")
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

    DS_NAME = f"ball_center_radius_patches_{COLOR_MODE.value.lower()}_{PATCH_TYPE.name.lower()}_border{BORDER}"
    DS_ROOT = f"../../data/{DS_NAME}/"

    NAOTH_SAVE_DIR = f"/tmp/naoth_labeled_patches_ball_center_radius_{PATCH_TYPE.name.lower()}_border{BORDER}"
    TK_SAVE_DIR = f"/tmp/tk03_combined_detection"
    TK_FILE_PATH = f"{TK_SAVE_DIR}/tk03_combined_detection.pkl"

    make_data_dir()
    download_patches(overwrite=args.force_download)

    # Currently we only do segmentation patches for the bottom camera
    if PATCH_TYPE == PatchType.LEGACY:
        print("\nCreating datasets combined")

        try:
            # TK03 dataset is only available in grayscale
            # and we don't have the top/bottom camera information
            if COLOR_MODE == ColorMode.YUV422_Y_ONLY_PIL:
                download_tk03_patches(overwrite=args.force_download)
                create_datasets_tk03_and_naoth_combined()
            else:
                # for color images we only have the naoth dataset
                # but we can split it into top and bottom patches
                create_datasets_combined()
        except Exception as e:
            print(f"Error creating combined datasets: {e}")

        try:
            print("\nCreating datasets top")
            create_datasets_top()
        except Exception as e:
            print(f"Error creating top datasets: {e}")

    try:
        print("\nCreating datasets bottom")
        create_datasets_bottom()
    except Exception as e:
        print(f"Error creating bottom datasets: {e}")
