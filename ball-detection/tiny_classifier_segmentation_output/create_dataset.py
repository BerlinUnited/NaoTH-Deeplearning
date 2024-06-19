import argparse
import os
import sys
import h5py
from tqdm import tqdm
import tempfile
import numpy as np
import cv2
from pathlib import Path
import random
from zipfile import ZipFile
import PIL.Image

helper_path = os.path.join(os.path.dirname(__file__), "../../tools")
sys.path.append(helper_path)

from helper import (
    get_postgres_cursor,
    get_minio_client,
    get_labelstudio_client,
    download_from_minio,
    load_image_as_yuv422_original,
)


def download_patches():
    camera = "bottom"
    sql_query = f"""SELECT bucket_{camera}_patches FROM robot_logs WHERE {camera}_validated = true"""
    print(sql_query)
    pg_cur = get_postgres_cursor()
    pg_cur.execute(sql_query)
    rtn_val = pg_cur.fetchall()
    data = [x[0] for x in rtn_val]

    mclient = get_minio_client()
    ls = get_labelstudio_client()

    # TODO create a output folder here
    Path("downloads").mkdir(exist_ok=True, parents=True)

    for bucket_name in sorted(data):
        print(f"Working on bucket {bucket_name}")
        file_path = download_from_minio(
            client=mclient, bucket_name=bucket_name, filename="patches_seg.zip", output_folder="downloads"
        )
        print(f"\t{file_path}")
        with ZipFile(file_path, "r") as f:
            f.extractall("datasets_segmentation")

        file_path = download_from_minio(
            client=mclient, bucket_name=bucket_name, filename="patches.zip", output_folder="downloads"
        )
        print(f"\t{file_path}")
        with ZipFile(file_path, "r") as f:
            f.extractall("datasets_old")


def make_classification_dataset_original():
    # TODO make a h5 file with two classes ball and not ball
    # in the first iteration dont care for the additional input
    ball_images = list(Path(f"./datasets_old/ball/").glob("**/*.png"))
    other_images = list(Path(f"./datasets_old/other/").glob("**/*.png"))
    penalty_images = list(Path(f"./datasets_old/penalty/").glob("**/*.png"))
    robot_images = list(Path(f"./datasets_old/robot/").glob("**/*.png"))

    full_list = list()
    full_list.extend(ball_images)
    full_list.extend(other_images)
    full_list.extend(penalty_images)
    full_list.extend(robot_images)
    random.shuffle(full_list)

    with h5py.File("training_ds_y_orig.h5", "w") as h5f:
        img_ds = h5f.create_dataset("X", shape=(len(full_list), 16, 16, 1), dtype=np.float32)
        label_ds = h5f.create_dataset("Y", shape=(len(full_list), 1), dtype=np.float32)
        for cnt, image_path in enumerate(tqdm(full_list)):
            image_yuv = load_image_as_yuv422_original(str(image_path))
            image_yuv = image_yuv.reshape(16, 16, 2)
            image_y = image_yuv[..., 0]
            image_y = np.array(image_y, dtype=np.uint8).reshape(16, 16, 1)
            image_y = image_y / 255.0
            # TODO try batching here for speedup
            img_ds[cnt : cnt + 1 :, :, :] = image_y
            if image_path in ball_images:
                label_ds[cnt : cnt + 1 :] = 1
            else:
                label_ds[cnt : cnt + 1 :] = 0


def make_classification_dataset_original_mean():
    # TODO make a h5 file with two classes ball and not ball
    # in the first iteration dont care for the additional input
    ball_images = list(Path(f"./datasets_old/ball/").glob("**/*.png"))
    other_images = list(Path(f"./datasets_old/other/").glob("**/*.png"))
    penalty_images = list(Path(f"./datasets_old/penalty/").glob("**/*.png"))
    robot_images = list(Path(f"./datasets_old/robot/").glob("**/*.png"))

    full_list = list()
    full_list.extend(ball_images)
    full_list.extend(other_images)
    full_list.extend(penalty_images)
    full_list.extend(robot_images)
    random.shuffle(full_list)

    with h5py.File("training_ds_y_orig_mean.h5", "w") as h5f:
        img_ds = h5f.create_dataset("X", shape=(len(full_list), 16, 16, 1), dtype=np.float32)
        label_ds = h5f.create_dataset("Y", shape=(len(full_list), 1), dtype=np.float32)
        for cnt, image_path in enumerate(tqdm(full_list)):
            image_yuv = load_image_as_yuv422_original(str(image_path))
            image_yuv = image_yuv.reshape(16, 16, 2)
            image_y = image_yuv[..., 0]
            image_y = np.array(image_y, dtype=np.uint8).reshape(16, 16, 1)
            image_y = image_y / 255.0
            # TODO try batching here for speedup
            img_ds[cnt : cnt + 1 :, :, :] = image_y
            if image_path in ball_images:
                label_ds[cnt : cnt + 1 :] = 1
            else:
                label_ds[cnt : cnt + 1 :] = 0

        mean = img_ds[:, :, 0].mean()
        img_ds[:, :, 0] -= mean


def make_fy1500_originaldata_y_meta_info():
    # TODO make a h5 file with two classes ball and not ball
    # in the first iteration dont care for the additional input
    ball_images = list(Path(f"./datasets_old/ball/").glob("**/*.png"))
    other_images = list(Path(f"./datasets_old/other/").glob("**/*.png"))
    penalty_images = list(Path(f"./datasets_old/penalty/").glob("**/*.png"))
    robot_images = list(Path(f"./datasets_old/robot/").glob("**/*.png"))

    full_list = list()
    full_list.extend(ball_images)
    full_list.extend(other_images)
    full_list.extend(penalty_images)
    full_list.extend(robot_images)
    random.shuffle(full_list)

    with h5py.File("fy1500_originaldata_y_meta_info.h5", "w") as h5f:
        img_ds = h5f.create_dataset("X", shape=(len(full_list), 16, 16, 2), dtype=np.float32)
        label_ds = h5f.create_dataset("Y", shape=(len(full_list), 1), dtype=np.float32)
        for cnt, image_path in enumerate(tqdm(full_list)):
            img = PIL.Image.open(image_path)
            p_min_x = int(img.info["p_min_x"]) / 640
            p_min_y = int(img.info["p_min_y"]) / 480
            p_max_x = int(img.info["p_max_x"]) / 640
            p_max_y = int(img.info["p_max_y"]) / 480
            patch_meta = np.array([[p_min_x, p_min_y], [p_max_x, p_max_y]])
            patch_meta = np.tile(patch_meta, (8, 8))

            image_yuv = load_image_as_yuv422_original(str(image_path))
            image_yuv = image_yuv.reshape(16, 16, 2)
            # image_y =  image_yuv[..., 0]
            image_y = np.array(image_yuv, dtype=np.uint8).reshape(16, 16, 2)
            image_y = image_y / 255.0
            image_y[:, :, 1] = patch_meta

            # TODO try batching here for speedup
            img_ds[cnt : cnt + 1 :, :, :] = image_y
            if image_path in ball_images:
                label_ds[cnt : cnt + 1 :] = 1
            else:
                label_ds[cnt : cnt + 1 :] = 0


def make_fy1500_originaldata_y_mean_subtracted_meta_info():
    # TODO make a h5 file with two classes ball and not ball
    # in the first iteration dont care for the additional input
    ball_images = list(Path(f"./datasets_old/ball/").glob("**/*.png"))
    other_images = list(Path(f"./datasets_old/other/").glob("**/*.png"))
    penalty_images = list(Path(f"./datasets_old/penalty/").glob("**/*.png"))
    robot_images = list(Path(f"./datasets_old/robot/").glob("**/*.png"))

    full_list = list()
    full_list.extend(ball_images)
    full_list.extend(other_images)
    full_list.extend(penalty_images)
    full_list.extend(robot_images)
    random.shuffle(full_list)

    with h5py.File("fy1500_originaldata_y_mean_subtracted_meta_info.h5", "w") as h5f:
        img_ds = h5f.create_dataset("X", shape=(len(full_list), 16, 16, 2), dtype=np.float32)
        label_ds = h5f.create_dataset("Y", shape=(len(full_list), 1), dtype=np.float32)
        for cnt, image_path in enumerate(tqdm(full_list)):
            img = PIL.Image.open(image_path)
            p_min_x = int(img.info["p_min_x"]) / 640
            p_min_y = int(img.info["p_min_y"]) / 480
            p_max_x = int(img.info["p_max_x"]) / 640
            p_max_y = int(img.info["p_max_y"]) / 480
            patch_meta = np.array([[p_min_x, p_min_y], [p_max_x, p_max_y]])
            patch_meta = np.tile(patch_meta, (8, 8))

            image_yuv = load_image_as_yuv422_original(str(image_path))
            image_yuv = image_yuv.reshape(16, 16, 2)
            # image_y =  image_yuv[..., 0]
            image_y = np.array(image_yuv, dtype=np.uint8).reshape(16, 16, 2)
            image_y = image_y / 255.0
            image_y[:, :, 1] = patch_meta

            # TODO try batching here for speedup
            img_ds[cnt : cnt + 1 :, :, :] = image_y
            if image_path in ball_images:
                label_ds[cnt : cnt + 1 :] = 1
            else:
                label_ds[cnt : cnt + 1 :] = 0

        mean = img_ds[:, :, 0].mean()
        img_ds[:, :, 0] -= mean


def make_classification_dataset():
    # TODO make a h5 file with two classes ball and not ball
    # in the first iteration dont care for the additional input
    ball_images = list(Path(f"./datasets_segmentation/ball/").glob("**/*.png"))
    other_images = list(Path(f"./datasets_segmentation/other/").glob("**/*.png"))
    penalty_images = list(Path(f"./datasets_segmentation/penalty/").glob("**/*.png"))
    robot_images = list(Path(f"./datasets_segmentation/robot/").glob("**/*.png"))

    full_list = list()
    full_list.extend(ball_images)
    full_list.extend(other_images)
    full_list.extend(penalty_images)
    full_list.extend(robot_images)
    random.shuffle(full_list)

    with h5py.File("training_ds_y.h5", "w") as h5f:
        img_ds = h5f.create_dataset("X", shape=(len(full_list), 16, 16, 1), dtype=np.float32)
        label_ds = h5f.create_dataset("Y", shape=(len(full_list), 1), dtype=np.float32)
        for cnt, image_path in enumerate(tqdm(full_list)):
            image_yuv = load_image_as_yuv422_original(str(image_path))
            image_yuv = image_yuv.reshape(16, 16, 2)
            image_y = image_yuv[..., 0]
            image_y = np.array(image_y, dtype=np.uint8).reshape(16, 16, 1)
            image_y = image_y / 255.0
            # TODO try batching here for speedup
            img_ds[cnt : cnt + 1 :, :, :] = image_y
            if image_path in ball_images:
                label_ds[cnt : cnt + 1 :] = 1
            else:
                label_ds[cnt : cnt + 1 :] = 0


def make_classification_dataset_mean():
    # TODO make a h5 file with two classes ball and not ball
    # in the first iteration dont care for the additional input
    ball_images = list(Path(f"./datasets_segmentation/ball/").glob("**/*.png"))
    other_images = list(Path(f"./datasets_segmentation/other/").glob("**/*.png"))
    penalty_images = list(Path(f"./datasets_segmentation/penalty/").glob("**/*.png"))
    robot_images = list(Path(f"./datasets_segmentation/robot/").glob("**/*.png"))

    full_list = list()
    full_list.extend(ball_images)
    full_list.extend(other_images)
    full_list.extend(penalty_images)
    full_list.extend(robot_images)
    random.shuffle(full_list)

    with h5py.File("training_ds_y_mean.h5", "w") as h5f:
        img_ds = h5f.create_dataset("X", shape=(len(full_list), 16, 16, 1), dtype=np.float32)
        label_ds = h5f.create_dataset("Y", shape=(len(full_list), 1), dtype=np.float32)
        for cnt, image_path in enumerate(tqdm(full_list)):
            image_yuv = load_image_as_yuv422_original(str(image_path))
            image_yuv = image_yuv.reshape(16, 16, 2)
            image_y = image_yuv[..., 0]
            image_y = np.array(image_y, dtype=np.uint8).reshape(16, 16, 1)
            image_y = image_y / 255.0
            # TODO try batching here for speedup
            img_ds[cnt : cnt + 1 :, :, :] = image_y
            if image_path in ball_images:
                label_ds[cnt : cnt + 1 :] = 1
            else:
                label_ds[cnt : cnt + 1 :] = 0

        mean = img_ds[:, :, 0].mean()
        img_ds[:, :, 0] -= mean


def make_classification_dataset_extended():
    # TODO make a h5 file with two classes ball and not ball
    # in the first iteration dont care for the additional input
    ball_images = list(Path(f"./datasets_segmentation/ball/").glob("**/*.png"))
    other_images = list(Path(f"./datasets_segmentation/other/").glob("**/*.png"))
    penalty_images = list(Path(f"./datasets_segmentation/penalty/").glob("**/*.png"))
    robot_images = list(Path(f"./datasets_segmentation/robot/").glob("**/*.png"))

    full_list = list()
    full_list.extend(ball_images)
    full_list.extend(other_images)
    full_list.extend(penalty_images)
    full_list.extend(robot_images)
    random.shuffle(full_list)

    with h5py.File("training_ds_meta_info.h5", "w") as h5f:
        img_ds = h5f.create_dataset("X", shape=(len(full_list), 16, 16, 2), dtype=np.float32)
        label_ds = h5f.create_dataset("Y", shape=(len(full_list), 1), dtype=np.float32)
        for cnt, image_path in enumerate(tqdm(full_list)):
            # load meta data first
            img = PIL.Image.open(image_path)
            p_min_x = int(img.info["p_min_x"]) / 640
            p_min_y = int(img.info["p_min_y"]) / 480
            p_max_x = int(img.info["p_max_x"]) / 640
            p_max_y = int(img.info["p_max_y"]) / 480
            patch_meta = np.array([[p_min_x, p_min_y], [p_max_x, p_max_y]])
            patch_meta = np.tile(patch_meta, (8, 8))

            image_yuv = load_image_as_yuv422_original(str(image_path))
            image_yuv = image_yuv.reshape(16, 16, 2)
            image_y = np.array(image_yuv, dtype=np.uint8).reshape(16, 16, 2)
            image_y = image_y / 255.0
            image_y[:, :, 1] = patch_meta

            # TODO try batching here for speedup
            img_ds[cnt : cnt + 1 :, :, :] = image_y

            if image_path in ball_images:
                label_ds[cnt : cnt + 1 :] = 1
            else:
                label_ds[cnt : cnt + 1 :] = 0


def make_classification_dataset_extended_mean():
    # TODO make a h5 file with two classes ball and not ball
    # in the first iteration dont care for the additional input
    ball_images = list(Path(f"./datasets_segmentation/ball/").glob("**/*.png"))
    other_images = list(Path(f"./datasets_segmentation/other/").glob("**/*.png"))
    penalty_images = list(Path(f"./datasets_segmentation/penalty/").glob("**/*.png"))
    robot_images = list(Path(f"./datasets_segmentation/robot/").glob("**/*.png"))

    full_list = list()
    full_list.extend(ball_images)
    full_list.extend(other_images)
    full_list.extend(penalty_images)
    full_list.extend(robot_images)
    random.shuffle(full_list)

    with h5py.File("training_ds_meta_info_mean.h5", "w") as h5f:
        img_ds = h5f.create_dataset("X", shape=(len(full_list), 16, 16, 2), dtype=np.float32)
        label_ds = h5f.create_dataset("Y", shape=(len(full_list), 1), dtype=np.float32)
        for cnt, image_path in enumerate(tqdm(full_list)):
            # load meta data first
            img = PIL.Image.open(image_path)
            p_min_x = int(img.info["p_min_x"]) / 640
            p_min_y = int(img.info["p_min_y"]) / 480
            p_max_x = int(img.info["p_max_x"]) / 640
            p_max_y = int(img.info["p_max_y"]) / 480
            patch_meta = np.array([[p_min_x, p_min_y], [p_max_x, p_max_y]])
            patch_meta = np.tile(patch_meta, (8, 8))

            image_yuv = load_image_as_yuv422_original(str(image_path))
            image_yuv = image_yuv.reshape(16, 16, 2)
            image_y = np.array(image_yuv, dtype=np.uint8).reshape(16, 16, 2)
            image_y = image_y / 255.0
            image_y[:, :, 1] = patch_meta

            # TODO try batching here for speedup
            img_ds[cnt : cnt + 1 :, :, :] = image_y

            if image_path in ball_images:
                label_ds[cnt : cnt + 1 :] = 1
            else:
                label_ds[cnt : cnt + 1 :] = 0

        mean = img_ds[:, :, 0].mean()
        img_ds[:, :, 0] -= mean


download_patches()
# make_classification_dataset_original()
# make_classification_dataset_original_mean()
# make_classification_dataset_original_extended_mean()
# make_classification_dataset()
# make_classification_dataset_mean()
# make_classification_dataset_extended()
# make_classification_dataset_extended_mean()

make_fy1500_originaldata_y_meta_info()
make_fy1500_originaldata_y_mean_subtracted_meta_info()
