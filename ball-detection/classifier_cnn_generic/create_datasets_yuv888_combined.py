import os
import sys

import h5py

helper_path = os.path.join(os.path.dirname(__file__), "../../tools")
sys.path.append(helper_path)

from helper import (
    ColorMode,
    combine_datasets_split_train_val_stratify,
    download_devils_labeled_patches,
    download_naoth_labeled_patches,
    get_classification_data_devils_combined,
    get_classification_data_naoth_combined,
)

if __name__ == "__main__":
    DEVILS_SAVE_DIR = "/tmp/devils_dataset"
    DEVILS_SAVE_DIR_EXTRACTED = (
        f"{DEVILS_SAVE_DIR}/patches_classification_naodevils_32x32x3_GO24"
    )
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

    print("Converting devils data to numpy arrays...")
    X_train_devils, y_train_devils = get_classification_data_devils_combined(
        file_path=DEVILS_SAVE_DIR_EXTRACTED,
        patch_size=(16, 16),
        color_mode=ColorMode.YUV888,
    )

    print("Converting naoth training data to numpy arrays...")
    X_train_naoth, y_train_naoth = get_classification_data_naoth_combined(
        file_path=NAOTH_TRAIN_SAVE_DIR,
        patch_size=(16, 16),
        color_mode=ColorMode.YUV888,
        filter_ambiguous_balls=True,
    )

    print("Merging devils and naoth training data with stratified splitting...")
    X_train, y_train, X_val, y_val = combine_datasets_split_train_val_stratify(
        Xs=[X_train_devils, X_train_naoth],
        ys=[y_train_devils, y_train_naoth],
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

    with h5py.File(
        "classification_patches_yuv888_devils+naoth_train_ball_no_ball_X_y.h5",
        "w",
    ) as f:
        f.create_dataset("X", data=X_train)
        f.create_dataset("y", data=y_train)

    with h5py.File(
        "classification_patches_yuv888_devils+naoth_val_ball_no_ball_X_y.h5", "w"
    ) as f:
        f.create_dataset("X", data=X_val)
        f.create_dataset("y", data=y_val)

    # Save devils training data separately
    with h5py.File(
        "classification_patches_yuv888_devils_ball_no_ball_X_y.h5",
        "w",
    ) as f:
        f.create_dataset("X", data=X_train_devils)
        f.create_dataset("y", data=y_train_devils)

    #############################################################################
    # Skip test data creation for now,                                    #######
    # the patches for the top buckets are not created yet (not validated) #######
    #############################################################################

    # print("Downloading naoth test data...")
    # download_naoth_labeled_patches(
    #     save_dir=NAOTH_TEST_SAVE_DIR,
    #     validated=False,
    #     filter_top=naoth_test_top_filter,
    #     filter_bottom=naoth_test_bottom_filter,
    # )

    # print("Converting naoth test data to numpy arrays...")
    # X_test, y_test = get_classification_data_naoth_combined(
    #     file_path=NAOTH_TEST_SAVE_DIR,
    #     patch_size=(16, 16),
    #     color_mode=ColorMode.YUV888,
    #     filter_ambiguous_balls=True,
    # )

    # print()
    # print("X_test shape:", X_test.shape)
    # print("y_test shape:", y_test.shape)
    # print("Number of ball examples in test set:", y_test.sum())
    # print()

    # with h5py.File(
    #     "classification_patches_yuv888_naoth_test_ball_no_ball_X_y.h5", "w"
    # ) as f:
    #     f.create_dataset("X", data=X_test)
    #     f.create_dataset("y", data=y_test)
