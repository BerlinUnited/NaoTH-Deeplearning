"""
    Creates a dataset based on data from labelstudio. The created dataset will be uploaded to datasets.naoth.de

    FIXME: eventually move downloading and h5 dataset creation to tools folder
    FIXME: its pretty slow
    FIXME: use middlepoint and gaussian distribution for ball and penalty mark similar to what bhuman did
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

helper_path = os.path.join(os.path.dirname(__file__), '../tools')
sys.path.append(helper_path)

from helper import get_postgres_cursor, get_minio_client, get_labelstudio_client, download_from_minio, load_image_as_yuv422, load_image_as_yuv422_y_only_better


def download_datasets(camera):
    sql_query = f"""SELECT log_path, ls_project_{camera}, bucket_{camera} FROM robot_logs WHERE {camera}_validated = true"""
    print(sql_query)
    pg_cur = get_postgres_cursor()
    pg_cur.execute(sql_query)
    rtn_val = pg_cur.fetchall()
    data = [x for x in rtn_val]

    mclient = get_minio_client()
    ls = get_labelstudio_client()

    def middle(n):  
        return n[1] 

    for logpath, ls_prj, bucket_name in sorted(data, key=middle):
        
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
                    
            
            if len(bbox_list_ball) > 0 or len(bbox_list_penalty) > 0 or len(bbox_list_robot) > 0:
                # image part
                image_file_name = task_output["storage_filename"]
                image_path = download_from_minio(client=mclient, bucket_name=bucket_name, filename=image_file_name, output_folder=download_folder)

                # label part 2
                img = cv2.imread(image_path)
                mask = np.zeros((img.shape[0],img.shape[1], 3),dtype=np.float32) # initialize mask

                for box in bbox_list_ball:
                    y_px, height_px, x_px, width_px = box
                    mask[y_px:y_px+height_px, x_px:x_px+width_px, 0] = 1.0

                for box in bbox_list_penalty:
                    y_px, height_px, x_px, width_px = box
                    mask[y_px:y_px+height_px,x_px:x_px+width_px, 1] = 1.0

                for box in bbox_list_robot:
                    y_px, height_px, x_px, width_px = box
                    mask[y_px:y_px+height_px,x_px:x_px+width_px, 2] = 1.0

                a = Path(download_folder) / "label"
                Path(a).mkdir(exist_ok=True, parents=True)
                mask_output_path = Path(download_folder) / "label" / str(bucket_name + "_" + image_file_name)
                cv2.imwrite(str(mask_output_path), mask)


def create_ds_y():
    images = list(Path("./all_labels/image").glob('**/*.png'))
    trainings_list = images[0:-100]
    validation_list = images[-100:]
    with h5py.File("training_ds_y.h5",'w') as h5f:
        img_ds = h5f.create_dataset('X',shape=(len(trainings_list), 240,320,1), dtype=np.float32)
        label_ds = h5f.create_dataset('Y',shape=(len(trainings_list), 15,20,1), dtype=np.float32)
        for cnt, image_path in enumerate(tqdm(trainings_list)):
            img = load_image_as_yuv422_y_only_better(str(image_path))  # FIXME
            print(img.shape)
            #img = cv2.resize(img, (320,240))
            # TODO try batching here for speedup
            img_ds[cnt:cnt+1:,:,:] = img

            label_path = image_path.parent.parent / "label" / image_path.name
            mask = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
            label_ds[cnt:cnt+1:,:,:] = mask
    with h5py.File("validation_ds_y.h5",'w') as h5f:
        img_ds = h5f.create_dataset('X',shape=(len(validation_list), 240,320,1), dtype=np.float32)
        label_ds = h5f.create_dataset('Y',shape=(len(validation_list), 15,20,1), dtype=np.float32)
        for cnt, image_path in enumerate(tqdm(validation_list)):
            img = load_image_as_yuv422_y_only_better(str(image_path))
            #img = cv2.resize(img, (320,240))
            # TODO try batching here for speedup
            img_ds[cnt:cnt+1:,:,:] = img

            label_path = image_path.parent.parent / "label" / image_path.name
            mask = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
            label_ds[cnt:cnt+1:,:,:] = mask

def create_ds_yuv():
    pass

def create_ds_rgb_all():
    images = list(Path("./all_labels/image").glob('**/*.png'))
    trainings_list = images[0:-100]
    validation_list = images[-100:]
    with h5py.File("training_ds_rgb.h5",'w') as h5f:
        img_ds = h5f.create_dataset('X',shape=(len(trainings_list), 240,320,3), dtype=np.float32)
        label_ds = h5f.create_dataset('Y',shape=(len(trainings_list), 15,20,3), dtype=np.float32)
        for cnt, image_path in enumerate(tqdm(trainings_list)):
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = cv2.resize(img, (320,240))
            # TODO try batching here for speedup
            img_ds[cnt:cnt+1:,:,:] = img

            label_path = image_path.parent.parent / "label" / image_path.name
            mask = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
            label_ds[cnt:cnt+1:,:,:] = mask

    with h5py.File("validation_ds_rgb.h5",'w') as h5f:
        img_ds = h5f.create_dataset('X',shape=(len(validation_list), 240,320,3), dtype=np.float32)
        label_ds = h5f.create_dataset('Y',shape=(len(validation_list), 15,20,3), dtype=np.float32)
        for cnt, image_path in enumerate(tqdm(validation_list)):
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = cv2.resize(img, (320,240))
            # TODO try batching here for speedup
            img_ds[cnt:cnt+1:,:,:] = img

            label_path = image_path.parent.parent / "label" / image_path.name
            mask = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
            label_ds[cnt:cnt+1:,:,:] = mask


def create_ds_gray_all():
    images = list(Path("./all_labels/image").glob('**/*.png'))
    trainings_list = images[0:-100]
    validation_list = images[-100:]
    with h5py.File("training_ds_gray_all.h5",'w') as h5f:
        img_ds = h5f.create_dataset('X',shape=(len(trainings_list), 240,320), dtype=np.float32)
        label_ds = h5f.create_dataset('Y',shape=(len(trainings_list), 15,20,3), dtype=np.float32)
        for cnt, image_path in enumerate(tqdm(trainings_list)):
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            img = img / 255.0
            img = cv2.resize(img, (320,240))
            # TODO try batching here for speedup
            img_ds[cnt:cnt+1:,:,:] = img

            label_path = image_path.parent.parent / "label" / image_path.name
            mask = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
            label_ds[cnt:cnt+1:,:,:] = mask

    with h5py.File("validation_ds_gray_all.h5",'w') as h5f:
        img_ds = h5f.create_dataset('X',shape=(len(validation_list), 240,320), dtype=np.float32)
        label_ds = h5f.create_dataset('Y',shape=(len(validation_list), 15,20,3), dtype=np.float32)
        for cnt, image_path in enumerate(tqdm(validation_list)):
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            img = img / 255.0
            img = cv2.resize(img, (320,240))
            # TODO try batching here for speedup
            img_ds[cnt:cnt+1:,:,:] = img

            label_path = image_path.parent.parent / "label" / image_path.name
            mask = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
            label_ds[cnt:cnt+1:,:,:] = mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", required=True, choices=['gray', 'yuv', 'rgb', 'y'])
    parser.add_argument("-c", "--camera", required=True, choices=['bottom', 'top'])
    args = parser.parse_args()

    download_datasets(args.camera)
    if args.type == "yuv":
        create_ds_yuv()
    if args.type == "gray":
        create_ds_gray_all()
    if args.type == "rgb":
        create_ds_rgb_all()
    if args.type == "y":
        create_ds_y()
