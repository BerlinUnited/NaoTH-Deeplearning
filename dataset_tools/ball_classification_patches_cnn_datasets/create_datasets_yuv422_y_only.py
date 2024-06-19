import os
import sys

import h5py
import numpy as np

helper_path = os.path.join(os.path.dirname(__file__), "../../tools")
sys.path.append(helper_path)

from tools.helper import (
    ColorMode,
    combine_datasets_split_train_val_stratify,
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


def create_yuv_422_combined_datasets():
    DS_BASE_NAME = f"{DS_ROOT}_combined_{PATCH_SIZE}x{PATCH_SIZE}"

    print("Converting devils data to numpy arrays...")
    X_devils, y_devils = get_classification_data_devils_combined(
        file_path=DEVILS_SAVE_DIR_EXTRACTED,
        color_mode=ColorMode.YUV422_Y_ONLY_PIL,
    )
    y_devils = np.array(y_devils)

    print("Converting naoth data to numpy arrays...")
    X_naoth, y_naoth = get_classification_data_naoth_combined(
        file_path=NAOTH_TRAIN_SAVE_DIR,
        color_mode=ColorMode.YUV422_Y_ONLY_PIL,
        filter_ambiguous_balls=True,
    )
    y_naoth = np.array(y_naoth)

    print(f"Resizing images to {PATCH_SIZE}x{PATCH_SIZE}...")
    X_devils = np.array([resize_image_cv2_inter_nearest(img, (PATCH_SIZE, PATCH_SIZE)) for img in X_devils])
    X_naoth = np.array([resize_image_cv2_inter_nearest(img, (PATCH_SIZE, PATCH_SIZE)) for img in X_naoth])

    print("Merging devils and naoth training data with stratified splitting...")
    X_train, y_train, X_val, y_val = combine_datasets_split_train_val_stratify(
        Xs=[X_devils, X_naoth],
        ys=[y_devils, y_naoth],
        test_size=0.15,
    )

    print()
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("Number of ball examples in training set:", y_train.sum())

    print()
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
    print("Number of ball examples in validation set:", y_val.sum())

    print("Saving datasets to HDF5 files...")

    with h5py.File(f"{DS_BASE_NAME}_devils+naoth_train_ball_no_ball_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_train)
        f.create_dataset("y", data=y_train)

    with h5py.File(f"{DS_BASE_NAME}_devils+naoth_val_ball_no_ball_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_val)
        f.create_dataset("y", data=y_val)

    # save devils data separately
    with h5py.File(f"{DS_BASE_NAME}_devils_ball_no_ball_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_devils)
        f.create_dataset("y", data=y_devils)

    # save naoth data separately
    with h5py.File(f"{DS_BASE_NAME}_naoth_ball_no_ball_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_naoth)
        f.create_dataset("y", data=y_naoth)

    #############################################################################
    # Skip test data creation for now,                                    #######
    # the patches for the top buckets are not created yet (not validated) #######
    #############################################################################

    # print("Converting naoth test data to numpy arrays...")
    # X_test, y_test = get_classification_data_naoth_combined(
    #     file_path=NAOTH_TEST_SAVE_DIR,
    #     color_mode=ColorMode.YUV422_Y_ONLY_PIL,
    #     filter_ambiguous_balls=True,
    # )

    # X_test = np.array([resize_image_cv2_inter_nearest(img, (PATCH_SIZE, PATCH_SIZE)) for img in X_test])
    # y_test = np.array(y_test)

    # print()
    # print("X_test shape:", X_test.shape)
    # print("y_test shape:", y_test.shape)
    # print("Number of ball examples in test set:", y_test.sum())
    # print()

    # with h5py.File(
    #     f"{DS_BASE_NAME}_naoth_test_ball_no_ball_X_y.h5", "w"
    # ) as f:
    #     f.create_dataset("X", data=X_test)
    #     f.create_dataset("y", data=y_test)


def create_yuv_422_top_datasets():
    DS_BASE_NAME = f"{DS_ROOT}_top_{PATCH_SIZE}x{PATCH_SIZE}"

    print("Converting devils data to numpy arrays...")
    X_devils, y_devils = get_classification_data_devils_top(
        file_path=DEVILS_SAVE_DIR_EXTRACTED,
        color_mode=ColorMode.YUV422_Y_ONLY_PIL,
    )
    y_devils = np.array(y_devils)

    print("Converting naoth data to numpy arrays...")
    X_naoth, y_naoth = get_classification_data_naoth_top(
        file_path=NAOTH_TRAIN_SAVE_DIR,
        color_mode=ColorMode.YUV422_Y_ONLY_PIL,
        filter_ambiguous_balls=True,
    )
    y_naoth = np.array(y_naoth)

    print(f"Resizing images to {PATCH_SIZE}x{PATCH_SIZE}...")
    X_devils = np.array([resize_image_cv2_inter_nearest(img, (PATCH_SIZE, PATCH_SIZE)) for img in X_devils])
    X_naoth = np.array([resize_image_cv2_inter_nearest(img, (PATCH_SIZE, PATCH_SIZE)) for img in X_naoth])

    print("Merging devils and naoth training data with stratified splitting...")
    X_train, y_train, X_val, y_val = combine_datasets_split_train_val_stratify(
        Xs=[X_devils, X_naoth],
        ys=[y_devils, y_naoth],
        test_size=0.15,
    )

    print()
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("Number of ball examples in training set:", y_train.sum())

    print()
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
    print("Number of ball examples in validation set:", y_val.sum())

    print("Saving datasets to HDF5 files...")

    with h5py.File(f"{DS_BASE_NAME}_devils+naoth_train_ball_no_ball_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_train)
        f.create_dataset("y", data=y_train)

    with h5py.File(f"{DS_BASE_NAME}_devils+naoth_val_ball_no_ball_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_val)
        f.create_dataset("y", data=y_val)

    # Save devils data separately
    with h5py.File(f"{DS_BASE_NAME}_devils_ball_no_ball_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_devils)
        f.create_dataset("y", data=y_devils)

    # save naoth data separately
    with h5py.File(f"{DS_BASE_NAME}_naoth_ball_no_ball_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_naoth)
        f.create_dataset("y", data=y_naoth)

    #############################################################################
    # Skip test data creation for now,                                    #######
    # the patches for the top buckets are not created yet (not validated) #######
    #############################################################################

    # print("Converting naoth test data to numpy arrays...")
    # X_test, y_test = get_classification_data_naoth_top(
    #     file_path=NAOTH_TEST_SAVE_DIR,
    #     color_mode=ColorMode.YUV422_Y_ONLY_PIL,
    #     filter_ambiguous_balls=True,
    # )

    # X_test = np.array([resize_image_cv2_inter_nearest(img, (PATCH_SIZE, PATCH_SIZE)) for img in X_test])
    # y_test = np.array(y_test)

    # print()
    # print("X_test shape:", X_test.shape)
    # print("y_test shape:", y_test.shape)
    # print("Number of ball examples in test set:", y_test.sum())
    # print()

    # with h5py.File(
    #     f"{DS_BASE_NAME}_naoth_test_ball_no_ball_X_y.h5", "w"
    # ) as f:
    #     f.create_dataset("X", data=X_test)
    #     f.create_dataset("y", data=y_test)


def create_yuv_422_bottom_datasets():
    DS_BASE_NAME = f"{DS_ROOT}_bottom_{PATCH_SIZE}x{PATCH_SIZE}"

    print("Converting devils data to numpy arrays...")
    X_devils, y_devils = get_classification_data_devils_bottom(
        file_path=DEVILS_SAVE_DIR_EXTRACTED,
        color_mode=ColorMode.YUV422_Y_ONLY_PIL,
    )
    y_devils = np.array(y_devils)

    print("Converting naoth training data to numpy arrays...")
    X_naoth, y_naoth = get_classification_data_naoth_bottom(
        file_path=NAOTH_TRAIN_SAVE_DIR,
        color_mode=ColorMode.YUV422_Y_ONLY_PIL,
        filter_ambiguous_balls=True,
    )
    y_naoth = np.array(y_naoth)

    print(f"Resizing images to {PATCH_SIZE}x{PATCH_SIZE}...")
    X_devils = np.array([resize_image_cv2_inter_nearest(img, (PATCH_SIZE, PATCH_SIZE)) for img in X_devils])
    X_naoth = np.array([resize_image_cv2_inter_nearest(img, (PATCH_SIZE, PATCH_SIZE)) for img in X_naoth])

    print("Merging devils and naoth training data with stratified splitting...")
    X_train, y_train, X_val, y_val = combine_datasets_split_train_val_stratify(
        Xs=[X_devils, X_naoth],
        ys=[y_devils, y_naoth],
        test_size=0.15,
    )

    print()
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("Number of ball examples in training set:", y_train.sum())

    print()
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
    print("Number of ball examples in validation set:", y_val.sum())

    print("Saving datasets to HDF5 files...")

    with h5py.File(f"{DS_BASE_NAME}_devils+naoth_train_ball_no_ball_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_train)
        f.create_dataset("y", data=y_train)

    with h5py.File(f"{DS_BASE_NAME}_devils+naoth_val_ball_no_ball_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_val)
        f.create_dataset("y", data=y_val)

    # Save devils data separately
    with h5py.File(f"{DS_BASE_NAME}_devils_ball_no_ball_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_devils)
        f.create_dataset("y", data=y_devils)

    # save naoth data separately
    with h5py.File(f"{DS_BASE_NAME}_naoth_ball_no_ball_X_y.h5", "w") as f:
        f.create_dataset("X", data=X_naoth)
        f.create_dataset("y", data=y_naoth)

    #############################################################################
    # Skip test data creation for now,                                    #######
    # the patches for the top buckets are not created yet (not validated) #######
    #############################################################################

    # print("Converting naoth test data to numpy arrays...")
    # X_test, y_test = get_classification_data_naoth_bottom(
    #     file_path=NAOTH_TEST_SAVE_DIR,
    #     color_mode=ColorMode.YUV422_Y_ONLY_PIL,
    #     filter_ambiguous_balls=True,
    # )

    # X_test = np.array([resize_image_cv2_inter_nearest(img, (PATCH_SIZE, PATCH_SIZE)) for img in X_test])
    # y_test = np.array(y_test)

    # print()
    # print("X_test shape:", X_test.shape)
    # print("y_test shape:", y_test.shape)
    # print("Number of ball examples in test set:", y_test.sum())
    # print()

    # with h5py.File(
    #     f"{DS_BASE_NAME}_naoth_test_ball_no_ball_X_y.h5", "w"
    # ) as f:
    #     f.create_dataset("X", data=X_test)
    #     f.create_dataset("y", data=y_test)


if __name__ == "__main__":
    PATCH_SIZE = 16
    DS_ROOT = "../data/classification_patches_yuv422_y_only"
    DEVILS_SAVE_DIR = "/tmp/devils_dataset"
    DEVILS_SAVE_DIR_EXTRACTED = f"{DEVILS_SAVE_DIR}/patches_classification_naodevils_32x32x3_GO24"
    NAOTH_TRAIN_SAVE_DIR = "/tmp/naoth_labeled_patches_train"
    NAOTH_TEST_SAVE_DIR = "/tmp/naoth_labeled_patches_test"

    naoth_train_top_filter = "ls_project_top NOT IN ('625', '626', '627', '628')"
    naoth_train_bottom_filter = "ls_project_bottom NOT IN ('629', '630', '631', '632')"
    naoth_test_top_filter = "ls_project_top IN ('625', '626', '627', '628')"
    naoth_test_bottom_filter = "ls_project_bottom IN ('629', '630', '631', '632')"

    print("Downloading devils data...")
    download_devils_labeled_patches(save_dir=DEVILS_SAVE_DIR)

    print("Downloading naoth training data...")
    download_naoth_labeled_patches(
        save_dir=NAOTH_TRAIN_SAVE_DIR,
        validated=True,
        filter_top=naoth_train_top_filter,
        filter_bottom=naoth_train_bottom_filter,
    )

    # print("Downloading naoth test data...")
    # download_naoth_labeled_patches(
    #     save_dir=NAOTH_TEST_SAVE_DIR,
    #     validated=False,
    #     filter_top=naoth_test_top_filter,
    #     filter_bottom=naoth_test_bottom_filter,
    # )

    create_yuv_422_combined_datasets()
    create_yuv_422_top_datasets()
    create_yuv_422_bottom_datasets()
