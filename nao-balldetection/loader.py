import csv
import json
import os
import random
import sys
from glob import glob
from os.path import relpath
from pathlib import Path
from PIL import Image as PIL_Image
import cv2
import numpy as np


def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def yuv888_bytes_to_yuv422_array(yuv888_bytes, width, height):
    yuv422 = np.ndarray(width * height * 2, np.uint8)

    for i in range(0, width * height, 2):
        yuv422[i * 2] = yuv888_bytes[i * 3]
        yuv422[i * 2 + 1] = (yuv888_bytes[i * 3 + 1] + yuv888_bytes[i * 3 + 4]) / 2.0
        yuv422[i * 2 + 2] = yuv888_bytes[i * 3 + 3]
        yuv422[i * 2 + 3] = (yuv888_bytes[i * 3 + 2] + yuv888_bytes[i * 3 + 5]) / 2.0

    return yuv422

def load_image_as_yuv422_pil(image_filename) -> np.ndarray:
    im = PIL_Image.open(image_filename)
    ycbcr = im.convert("YCbCr")
    #width, height = ycbcr.size

    #yuv422 = yuv888_bytes_to_yuv422_array(ycbcr.tobytes(), width=width, height=height)

    # cv2 size is (height, width)
    # Pillow size is (width, height)
    # we need to ensure consistent output shapes for all image loading functions
    #test = ycbcr.reshape(height, width, 3)
    #print(np.asarray(ycbcr).shape)
    #quit()
    return np.asarray(ycbcr)


def create_natural_classification_dataset(path, res):
    print("Loading images from " + path + " ...")
    db_balls = []
    db_noballs = []
    #print(path)
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}
    
    # load non ball images
    image_dir = Path(path) / "0.00"
    for image_path in image_dir.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in image_extensions:
            #print(image_path)
            img = load_image_as_yuv422_pil(image_path)
            img_normalized = img.astype(float) / 255.0
    
            #img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            #img = cv2.resize(img, (res["x"], res["y"]))
            #img = img * 1.3
            #img_normalized = img.astype(float) / 255.0

            target = np.array([0.0])
            db_noballs.append((img_normalized, target, image_path))
    
    image_dir = Path(path) / "1.00"
    for image_path in image_dir.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in image_extensions:
            #print(image_path)
            img = load_image_as_yuv422_pil(image_path)
            img_normalized = img.astype(float) / 255.0
            #img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            #img = cv2.resize(img, (res["x"], res["y"]))
            #img_normalized = img.astype(float) / 255.0

            target = np.array([1.0])
            db_balls.append((img_normalized, target, image_path))

    return db_balls, db_noballs

def create_natural_dataset(root_path, res, limit_noballs, dataset_type="detection"):
    #print("Looking for csv files in: ", root_path)
    complete_db_ball_list = []
    complete_db_noball_list = []

    all_paths = [root_path]

    # process files
    for path in all_paths:
        if dataset_type == "classification":
            db_ball_list, db_noball_list = create_natural_classification_dataset(str(path), res)
            complete_db_ball_list.extend(db_ball_list)
            complete_db_noball_list.extend(db_noball_list)
        elif dataset_type == "detection":
            db_ball_list, db_noball_list = create_natural_detection_dataset(str(path), res)
            complete_db_ball_list.extend(db_ball_list)
            complete_db_noball_list.extend(db_noball_list)
        elif dataset_type == "detection2":
            db_ball_list, db_noball_list = create_natural_detection_dataset_without_classification(str(path), res)
            complete_db_ball_list.extend(db_ball_list)
            complete_db_noball_list.extend(db_noball_list)
        elif dataset_type == "segmentation":
            db_ball_list, db_noball_list = create_natural_segmentation_dataset(str(path), res)
            complete_db_ball_list.extend(db_ball_list)
            complete_db_noball_list.extend(db_noball_list)
        else:
            print("ERROR: unsupported dataset type")
            sys.exit()
    print("len db_ball", len(complete_db_ball_list))
    print("len db_noball_list", len(complete_db_noball_list))
    if limit_noballs is True and len(complete_db_ball_list) < len(complete_db_noball_list):
        #print("Limit negative images to ", len(complete_db_ball_list))
        no_ball_mask = np.random.choice(len(complete_db_noball_list), len(complete_db_ball_list))
        complete_db_noball_list = [complete_db_noball_list[i] for i in no_ball_mask]

    db = complete_db_ball_list + complete_db_noball_list
    random.shuffle(db)
    #print(db)
  
    input_images, targets, file_paths = list(map(np.array, list(zip(*db))))

    # expand dimensions of the input images for use with tensorflow
    #input_images = input_images.reshape(*input_images.shape, 1)

    print("Loading finished")
    print("\nStatistic:")
    print("number of images: " + str(len(input_images)) + " balls images: " + str(len(complete_db_ball_list)) +
          " no ball images: " + str(len(complete_db_noball_list)))

    return input_images, targets, file_paths

def calculate_mean(images):
    print(images.shape)
    if images.shape[3] == 3:
        return np.mean(images, axis=(0, 1, 2))

    return np.mean(images)

def subtract_mean(images, mean):
    return images - mean
