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
import h5py
from tqdm import tqdm
import tempfile
import numpy as np
import cv2
from pathlib import Path
from naoth.log import BoundingBox, Point2D

helper_path = os.path.join(os.path.dirname(__file__), '../tools')
sys.path.append(helper_path)

from helper import get_postgres_cursor, get_minio_client, get_labelstudio_client, download_from_minio, load_image_as_yuv422, load_image_as_yuv422_y_only_better


def download_datasets(camera, grid_size):
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

        download_folder = Path("./datasets") / camera
        Path(download_folder).mkdir(exist_ok=True, parents=True)
        for task_output in tasks:
            # label part 1
            bbox_list_robot = list()
            bbox_list_penalty = list()
            bbox_list_ball = list()
            for anno in task_output['annotations']:
                results = anno["result"]
                for result in results:
                    # ignore relations here
                    if result["type"] != "rectanglelabels":
                        continue
                    actual_label = result["value"]["rectanglelabels"][0]

                    # x,y,width,height are all percentages within [0,100]
                    x, y, width, height = result["value"]["x"], result["value"]["y"], result["value"]["width"], result["value"]["height"]
                    img_width = result['original_width']
                    img_height = result['original_height']
                    # FIXME int might not be the best rounding method here - but off by one pixel is also not that bad
                    x_px = int(x / 100 * img_width)
                    y_px = int(y / 100 * img_height)
                    width_px = int(width / 100 * img_width)
                    height_px = int(height / 100 * img_height)

                    if actual_label == "ball":
                        bbox_list_ball.append((y_px,height_px,x_px,width_px))
                    if actual_label == "penalty_mark":
                        bbox_list_penalty.append((y_px,height_px,x_px,width_px))
                    if actual_label == "nao":
                        bbox_list_robot.append((y_px,height_px,x_px,width_px))
                    
            # TODO creating the masks makes it harder to calculate overlap with grid cells later so do it here and the output will then always be in the grid shape and not the image shape
            # TODO figure out how a better workflow that also includes the adjustments to the robot masks could work -> after robocup and with new labeltool
            if len(bbox_list_ball) > 0 or len(bbox_list_penalty) > 0 or len(bbox_list_robot) > 0:
                # image part
                image_file_name = task_output["storage_filename"]
                image_path = download_from_minio(client=mclient, bucket_name=bucket_name, filename=image_file_name, output_folder=download_folder)

                # label part 2
                img = cv2.imread(image_path)
                img_height = img.shape[0]
                img_width = img.shape[1]

                grid_rows, grid_columns = grid_size  # FIXME better names its not actually height but num cols num rows or something like that
                grid_cell_height = int(img_height / grid_rows)
                grid_cell_width = int(img_width / grid_columns)

                # use defined grid shape here
                mask = np.zeros((grid_rows,grid_columns, 3),dtype=np.float32) # initialize mask
 
                for box in bbox_list_ball:
                    y_px, height_px, x_px, width_px = box
                    ball_bb = BoundingBox.from_xywh(x_px, y_px,width_px, height_px)
                    for y in range(grid_rows):
                        for x in range(grid_columns):
                                cell_x1 = x * grid_cell_width
                                cell_y1 = y * grid_cell_height
                                cell_x2 = x * grid_cell_width + grid_cell_width
                                cell_y2 = y * grid_cell_height + grid_cell_height
                                cell_bb = BoundingBox.from_coords(cell_x1, cell_y1,cell_x2, cell_y2)
                                intersection = cell_bb.intersection(ball_bb)
                                if not intersection is None:
                                    value = intersection.area / cell_bb.area

                                    mask[y,x, 0] = value * 255.0 # because png can only handle ints argh

                for box in bbox_list_penalty:
                    y_px, height_px, x_px, width_px = box
                    # TODO put this in an extra function
                    penalty_bb = BoundingBox.from_xywh(x_px, y_px,width_px, height_px)
                    for y in range(grid_rows):
                        for x in range(grid_columns):
                                cell_x1 = x * grid_cell_width
                                cell_y1 = y * grid_cell_height
                                cell_x2 = x * grid_cell_width + grid_cell_width
                                cell_y2 = y * grid_cell_height + grid_cell_height
                                cell_bb = BoundingBox.from_coords(cell_x1, cell_y1,cell_x2, cell_y2)
                                intersection = cell_bb.intersection(penalty_bb)
                                if not intersection is None:
                                    value = intersection.area / cell_bb.area
                                    mask[y,x, 1] = value * 255.0 # because png can only handle ints argh

                for box in bbox_list_robot:
                    y_px, height_px, x_px, width_px = box
                    # TODO put this in an extra function
                    robot_bb = BoundingBox.from_xywh(x_px, y_px,width_px, height_px)
                    for y in range(grid_rows):
                        for x in range(grid_columns):
                                cell_x1 = x * grid_cell_width
                                cell_y1 = y * grid_cell_height
                                cell_x2 = x * grid_cell_width + grid_cell_width
                                cell_y2 = y * grid_cell_height + grid_cell_height
                                cell_bb = BoundingBox.from_coords(cell_x1, cell_y1,cell_x2, cell_y2)
                                intersection = cell_bb.intersection(robot_bb)
                                if not intersection is None:
                                    value = intersection.area / cell_bb.area
                                    mask[y,x, 2] = value * 255.0 # because png can only handle ints argh

                # maybe use different output folders for different grid sizes?
                a = Path(download_folder) / "label"
                Path(a).mkdir(exist_ok=True, parents=True)
                mask_output_path = Path(download_folder) / "label" / str(bucket_name + "_" + image_file_name)
                cv2.imwrite(str(mask_output_path), mask)


def create_ds_y(camera, scale_factor, ball_only=False):
    # FIXME use new folder structure
    # FIXME put validation and trainings set together
    # TODO use scale factor

    new_img_width = 640 / scale_factor
    new_img_height = 640 / scale_factor


    images = list(Path(f"./datasets/{camera}/image").glob('**/*.png'))

    trainings_list = images[0:-100]
    validation_list = images[-100:]
    with h5py.File("training_ds_y.h5",'w') as h5f:
        img_ds = h5f.create_dataset('X',shape=(len(trainings_list), new_img_height, new_img_width,1), dtype=np.float32)
        if ball_only:
            label_ds = h5f.create_dataset('Y',shape=(len(trainings_list), 15,20,1), dtype=np.float32)
        else:
            label_ds = h5f.create_dataset('Y',shape=(len(trainings_list), 15,20,3), dtype=np.float32)
        for cnt, image_path in enumerate(tqdm(trainings_list)):
            img = load_image_as_yuv422_y_only_better(str(image_path))  # FIXME
            # TODO try batching here for speedup
            img_ds[cnt:cnt+1:,:,:] = img

            label_path = image_path.parent.parent / "label" / image_path.name
            mask = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
            mask = mask / 255.0
            label_ds[cnt:cnt+1:,:,:] = mask

    with h5py.File("validation_ds_y.h5",'w') as h5f:
        img_ds = h5f.create_dataset('X',shape=(len(validation_list), new_img_height,new_img_width,1), dtype=np.float32)
        if ball_only:
            label_ds = h5f.create_dataset('Y',shape=(len(validation_list), 15,20,1), dtype=np.float32)
        else:
            label_ds = h5f.create_dataset('Y',shape=(len(validation_list), 15,20,3), dtype=np.float32)
        for cnt, image_path in enumerate(tqdm(validation_list)):
            img = load_image_as_yuv422_y_only_better(str(image_path))
            # TODO try batching here for speedup
            img_ds[cnt:cnt+1:,:,:] = img

            label_path = image_path.parent.parent / "label" / image_path.name
            mask = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
            mask = mask / 255.0
            if ball_only:
                mask = mask[:,:,0]
            label_ds[cnt:cnt+1:,:,:] = mask

def create_ds_yuv():
    pass


if __name__ == "__main__":
    # FIXME add upload to datasets.naoth.de
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", required=True, choices=['yuv', 'y'])
    parser.add_argument("-c", "--camera", required=True, choices=['bottom', 'top'])
    parser.add_argument("-g", "--grid", required=True, nargs=2, type=int, help="Set the grid size like this: -g #rows #cols")
    parser.add_argument("-s", "--scale_factor", required=False, type=int, default=2, help="The factor by which the image will be downscaled")

    args = parser.parse_args()
    # python create_dataset.py -t y -c bottom -g 15 20
    grid_size = tuple(args.grid)

    download_datasets(args.camera, grid_size)
    if args.type == "yuv":
        create_ds_yuv()
    if args.type == "y":
        create_ds_y(args.camera, args.scale_factor)
