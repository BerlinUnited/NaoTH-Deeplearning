import csv
import json
import os
import random
import sys
from glob import glob
from os.path import relpath
from pathlib import Path

import cv2
import numpy as np


def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def create_natural_classification_dataset(path, res):
    print("Loading images from " + path + " ...")
    db_balls = []
    db_noballs = []
    #print(path)
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}
    
    # load non ball images
    image_dir = Path(path) / "0"
    for image_path in image_dir.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in image_extensions:
            #print(image_path)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (res["x"], res["y"]))
            img = img * 1.3
            img_normalized = img.astype(float) / 255.0

            target = np.array([0.0])
            db_noballs.append((img_normalized, target, image_path))
    
    image_dir = Path(path) / "1"
    for image_path in image_dir.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in image_extensions:
            #print(image_path)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (res["x"], res["y"]))
            img_normalized = img.astype(float) / 255.0

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
    input_images = input_images.reshape(*input_images.shape, 1)

    print("Loading finished")
    print("\nStatistic:")
    print("number of images: " + str(len(input_images)) + " balls images: " + str(len(complete_db_ball_list)) +
          " no ball images: " + str(len(complete_db_noball_list)))

    return input_images, targets, file_paths


def calculate_mean(images):
    return np.mean(images)


def subtract_mean(images, mean):
    return images - mean
