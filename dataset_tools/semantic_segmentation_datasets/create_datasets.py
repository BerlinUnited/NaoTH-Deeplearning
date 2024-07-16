"""
    Creates a dataset based on data from labelstudio. The created dataset will be uploaded to datasets.naoth.de

    FIXME: eventually move downloading and h5 dataset creation to tools folder
    FIXME: add augmentation 
    FIXME: make yuv422 work
    FIXME: better validation split
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np
from naoth.log import BoundingBox
from tqdm import tqdm

helper_path = os.path.join(os.path.dirname(__file__), "../../")
sys.path.append(helper_path)

from tools import (
    ColorMode,
    download_from_minio,
    get_labelstudio_client,
    get_minio_client,
    get_postgres_cursor,
    load_image_as_yuv422_y_only_pil,
    load_image_as_yuv422_pil,
    str2bool
)


def download_images_and_masks(output_folder, camera, grid_size):
    sql_query = f"""SELECT ls_project_{camera}, bucket_{camera} FROM robot_logs WHERE {camera}_validated = true"""

    print(sql_query)
    pg_cur = get_postgres_cursor()
    pg_cur.execute(sql_query)
    rtn_val = pg_cur.fetchall()
    data = [x for x in rtn_val]

    mclient = get_minio_client()
    ls = get_labelstudio_client()

    for ls_prj, bucket_name in sorted(data):

        print(f"Working on project {ls_prj}")

        project = ls.get_project(ls_prj)
        tasks = project.get_labeled_tasks()

        # TODO download all files for a project
        # TODO move the files inside an h5 file in the current folder
        # TODO think about structure inside h5 file
        # TODO first draft I can make mask from bounding boxes like this: https://stackoverflow.com/questions/64195636/converting-bounding-box-regions-into-masks-and-saving-them-as-png-files

        download_folder = Path(output_folder)
        download_folder_images = download_folder / "images"
        Path(download_folder).mkdir(exist_ok=True, parents=True)
        for task_output in tasks:
            # label part 1
            bbox_list_robot = list()
            bbox_list_penalty = list()
            bbox_list_ball = list()
            for anno in task_output["annotations"]:
                results = anno["result"]
                for result in results:
                    # ignore relations here
                    if result["type"] != "rectanglelabels":
                        continue
                    actual_label = result["value"]["rectanglelabels"][0]

                    # x,y,width,height are all percentages within [0,100]
                    x, y, width, height = (
                        result["value"]["x"],
                        result["value"]["y"],
                        result["value"]["width"],
                        result["value"]["height"],
                    )
                    img_width = result["original_width"]
                    img_height = result["original_height"]
                    # FIXME int might not be the best rounding method here - but off by one pixel is also not that bad
                    x_px = int(x / 100 * img_width)
                    y_px = int(y / 100 * img_height)
                    width_px = int(width / 100 * img_width)
                    height_px = int(height / 100 * img_height)

                    if actual_label == "ball":
                        bbox_list_ball.append((y_px, height_px, x_px, width_px))
                    if actual_label == "penalty_mark":
                        bbox_list_penalty.append((y_px, height_px, x_px, width_px))
                    if actual_label == "nao":
                        bbox_list_robot.append((y_px, height_px, x_px, width_px))

            # TODO creating the masks makes it harder to calculate overlap with grid cells later so do it here and the output will then always be in the grid shape and not the image shape
            # TODO figure out how a better workflow that also includes the adjustments to the robot masks could work -> after robocup and with new labeltool
            if len(bbox_list_ball) > 0 or len(bbox_list_penalty) > 0 or len(bbox_list_robot) > 0:
                # image part
                image_file_name = task_output["storage_filename"]
                image_path = download_from_minio(
                    client=mclient,
                    bucket_name=bucket_name,
                    filename=image_file_name,
                    output_folder=download_folder_images,
                )

                # label part 2
                img = cv2.imread(image_path)
                img_height = img.shape[0]
                img_width = img.shape[1]

                grid_rows, grid_columns = (
                    grid_size  # FIXME better names its not actually height but num cols num rows or something like that
                )
                grid_cell_height = int(img_height / grid_rows)
                grid_cell_width = int(img_width / grid_columns)

                # use defined grid shape here
                mask = np.zeros((grid_rows, grid_columns, 3), dtype=np.float32)  # initialize mask

                for box in bbox_list_ball:
                    y_px, height_px, x_px, width_px = box
                    ball_bb = BoundingBox.from_xywh(x_px, y_px, width_px, height_px)
                    for y in range(grid_rows):
                        for x in range(grid_columns):
                            cell_x1 = x * grid_cell_width
                            cell_y1 = y * grid_cell_height
                            cell_x2 = x * grid_cell_width + grid_cell_width
                            cell_y2 = y * grid_cell_height + grid_cell_height
                            cell_bb = BoundingBox.from_coords(cell_x1, cell_y1, cell_x2, cell_y2)
                            intersection = cell_bb.intersection(ball_bb)
                            if not intersection is None:
                                value = intersection.area / cell_bb.area

                                mask[y, x, 0] = value * 255.0  # because png can only handle ints argh

                for box in bbox_list_penalty:
                    y_px, height_px, x_px, width_px = box
                    # TODO put this in an extra function
                    penalty_bb = BoundingBox.from_xywh(x_px, y_px, width_px, height_px)
                    for y in range(grid_rows):
                        for x in range(grid_columns):
                            cell_x1 = x * grid_cell_width
                            cell_y1 = y * grid_cell_height
                            cell_x2 = x * grid_cell_width + grid_cell_width
                            cell_y2 = y * grid_cell_height + grid_cell_height
                            cell_bb = BoundingBox.from_coords(cell_x1, cell_y1, cell_x2, cell_y2)
                            intersection = cell_bb.intersection(penalty_bb)
                            if not intersection is None:
                                value = intersection.area / cell_bb.area
                                mask[y, x, 1] = value * 255.0  # because png can only handle ints argh

                for box in bbox_list_robot:
                    y_px, height_px, x_px, width_px = box
                    # TODO put this in an extra function
                    robot_bb = BoundingBox.from_xywh(x_px, y_px, width_px, height_px)
                    for y in range(grid_rows):
                        for x in range(grid_columns):
                            cell_x1 = x * grid_cell_width
                            cell_y1 = y * grid_cell_height
                            cell_x2 = x * grid_cell_width + grid_cell_width
                            cell_y2 = y * grid_cell_height + grid_cell_height
                            cell_bb = BoundingBox.from_coords(cell_x1, cell_y1, cell_x2, cell_y2)
                            intersection = cell_bb.intersection(robot_bb)
                            if not intersection is None:
                                value = intersection.area / cell_bb.area
                                mask[y, x, 2] = value * 255.0  # because png can only handle ints argh

                # maybe use different output folders for different grid sizes?
                a = Path(download_folder) / "label"
                Path(a).mkdir(exist_ok=True, parents=True)
                mask_output_path = Path(download_folder) / "label" / str(bucket_name + "_" + image_file_name)
                cv2.imwrite(str(mask_output_path), mask)


def create_datasets(TMP_ROOT, DS_ROOT, DS_NAME, camera, scale_factor, color_mode, ball_only):
    CHANNELS = 1 if color_mode == ColorMode.YUV422_Y_ONLY_PIL else 2
    new_img_width = 640 / scale_factor
    new_img_height = 480 / scale_factor

    images = list(Path(f"{TMP_ROOT}/images").glob("**/*.png"))

    trainings_list = images[0:-100]
    validation_list = images[-100:]
    with h5py.File(Path(DS_ROOT) / (str(DS_NAME) + f"_{camera}_training.h5"), "w") as h5f:
        img_ds = h5f.create_dataset(
            "X",
            shape=(len(trainings_list), new_img_height, new_img_width, CHANNELS),
            dtype=np.float32,
        )
        if ball_only:
            label_ds = h5f.create_dataset("Y", shape=(len(trainings_list), 15, 20, 1), dtype=np.float32)
        else:
            label_ds = h5f.create_dataset("Y", shape=(len(trainings_list), 15, 20, 3), dtype=np.float32)
        for cnt, image_path in enumerate(tqdm(trainings_list)):
            if color_mode == ColorMode.YUV422_Y_ONLY_PIL:
                img = load_image_as_yuv422_y_only_pil(str(image_path))
            else:
                img = load_image_as_yuv422_pil(str(image_path))
            img = img / 255.0  # rescale to [0,1]
            img = img[::scale_factor, ::scale_factor]  # subsample by factor 2

            # TODO try batching here for speedup
            img_ds[cnt : cnt + 1 :, :, :] = img

            # FIXME use different folders for different grid sizes
            label_path = image_path.parent.parent / "label" / image_path.name
            mask = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
            mask = mask / 255.0

            if ball_only:
                label_ds[cnt : cnt + 1 :, :, :] = np.expand_dims(mask[:,:,0], axis=2) # we only want the first channel but keep the dimension
            else:
                label_ds[cnt : cnt + 1 :, :, :] = mask

    with h5py.File(Path(DS_ROOT) / (str(DS_NAME) + f"_{camera}_validation.h5"), "w") as h5f:
        img_ds = h5f.create_dataset(
            "X",
            shape=(len(validation_list), new_img_height, new_img_width, CHANNELS),
            dtype=np.float32,
        )
        if ball_only:
            label_ds = h5f.create_dataset("Y", shape=(len(validation_list), 15, 20, 1), dtype=np.float32)
        else:
            label_ds = h5f.create_dataset("Y", shape=(len(validation_list), 15, 20, 3), dtype=np.float32)
        for cnt, image_path in enumerate(tqdm(validation_list)):
            if color_mode == ColorMode.YUV422_Y_ONLY_PIL:
                img = load_image_as_yuv422_y_only_pil(str(image_path))
            else:
                img = load_image_as_yuv422_pil(str(image_path))
            img = img / 255.0  # rescale to [0,1]
            img = img[::scale_factor, ::scale_factor]  # subsample by factor 2
            # TODO try batching here for speedup
            img_ds[cnt : cnt + 1 :, :, :] = img

            # FIXME use different folders for different grid sizes
            label_path = image_path.parent.parent / "label" / image_path.name
            mask = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
            mask = mask / 255.0
            if ball_only:
                label_ds[cnt : cnt + 1 :, :, :] = np.expand_dims(mask[:,:,0], axis=2) # we only want the first channel but keep the dimension
            else:
                label_ds[cnt : cnt + 1 :, :, :] = mask



def parse_args():
    parser = argparse.ArgumentParser()
    # fmt: off
    #parser.add_argument("-t", "--type", required=True, choices=["yuv", "y"])
    parser.add_argument("-c", "--camera", required=True, choices=["bottom", "top"])
    parser.add_argument("-g", "--grid", required=True, nargs=2, type=int, help="Set the grid size like this: -g #rows #cols")
    parser.add_argument("-s", "--scale_factor", required=False, type=int, default=2, help="The factor by which the image will be downscaled")
    parser.add_argument("--color_mode", type=ColorMode, choices=list(ColorMode), default=ColorMode.YUV422_Y_ONLY_PIL, help="Color mode")
    parser.add_argument("--ball_only", type=str2bool, help="output is only one class if true")
    # fmt: on

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # FIXME add upload to datasets.naoth.de
    args = parse_args()
    print("CREATING SEGMENTATION PATCHES DATASETS")
    print("========================================")
    print(f"SCALE_FACTOR = {args.scale_factor}")
    print(f"GRID_SIZE = {args.grid}")
    print(f"CAMERA = {args.camera}")
    print(f"COLOR_MODE = {args.color_mode}")
    print(f"BALL ONLY = {args.ball_only}")
    
    grid_size = tuple(args.grid)

    # folder for final datasets
    DS_NAME = f"segmentation_data_{args.color_mode.value.lower()}_grid_{grid_size}_input_size_{int(640/args.scale_factor)}x{int(480/args.scale_factor)}_ball_only_{args.ball_only}"
    DS_ROOT = f"../../data/{DS_NAME}/"
    Path(f"{DS_ROOT}").mkdir(parents=True, exist_ok=True)
    
    # folder for fullsize images and masks (temp data)
    TMP_DATA = f"segmentation_data_grid_{grid_size}_{args.camera}"
    TMP_ROOT = f"../../data/{TMP_DATA}/"
    Path(f"{TMP_ROOT}").mkdir(parents=True, exist_ok=True)

    #download_images_and_masks(TMP_ROOT, args.camera, grid_size)

    create_datasets(TMP_ROOT, DS_ROOT, DS_NAME, args.camera, args.scale_factor, args.color_mode, args.ball_only)