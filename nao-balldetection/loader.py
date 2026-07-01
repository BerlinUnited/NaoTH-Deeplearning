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


def create_natural_detection_dataset(path, res):
    print("Loading images from " + path + " ...")
    db_balls = []
    db_noballs = []

    # parse csv file
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            f = os.path.join(os.path.dirname(path), row["filename"])
            p = row["filename"]

            # load image
            try:
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (res["x"], res["y"]))
                img_normalized = img.astype(float) / 255.0
            except Exception as ex:
                print("Error loading image ", f)
                continue

            is_ball = False
            # load ball information
            region_count = int(row["region_count"])
            if region_count > 0:
                atts = json.loads(row["region_attributes"])
                if atts["type"] == "smudged_ball":
                    # ignore this image
                    continue
                elif atts["type"] == "ball":
                    shape = json.loads(row["region_shape_attributes"])
                    if shape["name"] == "circle":
                        x_coord = int(shape["cx"])
                        y_coord = int(shape["cy"])
                        radius = int(shape["r"])

                        # draw detected circle into debug image
                        # cv2.circle(debug_img, (int(x),int(y)), int(radius), color=(0,0,255))

                        # normalize to resolution
                        x_coord = (x_coord / res["x"])
                        y_coord = (y_coord / res["y"])
                        radius = radius / max(res["x"], res["y"])
                        is_ball = True
                    else:
                        # we only support circles
                        print("WARNING: Annotation is not a circle")
                        continue
                elif atts["type"] == "smudge":
                    continue
                else:
                    # unknown type
                    print("Unknown type \"" + atts["type"] + "\" in file " + f)
                    continue
            else:
                # no region means no ball
                radius = 0.0
                x_coord = 0.5
                y_coord = 0.5

            # for each row add the image and the prediction
            if is_ball:
                target = np.array([radius, x_coord, y_coord, 1.0])
                db_balls.append((img_normalized, target, p))
            else:
                target = np.array([radius, x_coord, y_coord, 0.0])
                db_noballs.append((img_normalized, target, p))

            # augment: binarized image
            bin_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2).reshape((16, 16))

            if is_ball:
                db_balls.append((bin_img.astype(float) / 255.0, target, p))
            else:
                db_noballs.append((bin_img.astype(float) / 255.0, target, p))

            # augment: gamma adjusted image
            # TODO do the augmentation after the image db creation in a extra script
            avg_img_f = np.average(img_normalized)
            if 0.2 <= avg_img_f <= 0.8:
                # augment: gamma
                for g in (0.4, 1.3):
                    if is_ball:
                        db_balls.append((adjust_gamma(img, g).astype(float) / 255.0, target, p))
                    else:
                        db_noballs.append((adjust_gamma(img, g).astype(float) / 255.0, target, p))

        print("len db_balls:", len(db_balls))
    return db_balls, db_noballs


def create_natural_detection_dataset_without_classification(path, res):
    """
    without classification value in output
    """
    print("Loading images from " + path + " ...")
    db_balls = []
    db_noballs = []

    # parse csv file
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            f = os.path.join(os.path.dirname(path), row["filename"])
            p = row["filename"]

            # load image
            try:
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (res["x"], res["y"]))
                img_normalized = img.astype(float) / 255.0
            except Exception as ex:
                print("Error loading image ", f)
                continue

            is_ball = False
            # load ball information
            region_count = int(row["region_count"])
            if region_count > 0:
                atts = json.loads(row["region_attributes"])
                if atts["type"] == "smudged_ball":
                    # ignore this image
                    continue
                elif atts["type"] == "ball":
                    shape = json.loads(row["region_shape_attributes"])
                    if shape["name"] == "circle":
                        x_coord = int(shape["cx"])
                        y_coord = int(shape["cy"])
                        radius = int(shape["r"])

                        # draw detected circle into debug image
                        # cv2.circle(debug_img, (int(x),int(y)), int(radius), color=(0,0,255))

                        # normalize to resolution
                        x_coord = (x_coord / res["x"])
                        y_coord = (y_coord / res["y"])
                        radius = radius / max(res["x"], res["y"])
                        is_ball = True
                    else:
                        # we only support circles
                        print("WARNING: Annotation is not a circle")
                        continue
                elif atts["type"] == "smudge":
                    continue
                else:
                    # unknown type
                    print("Unknown type \"" + atts["type"] + "\" in file " + f)
                    continue
            else:
                # no region means no ball
                radius = 0.0
                x_coord = 0.5
                y_coord = 0.5

            # for each row add the image and the prediction
            if is_ball:
                target = np.array([radius, x_coord, y_coord])
                db_balls.append((img_normalized, target, p))
            else:
                target = np.array([radius, x_coord, y_coord])
                db_noballs.append((img_normalized, target, p))

            # augment: binarized image
            bin_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2).reshape((16, 16))

            if is_ball:
                db_balls.append((bin_img.astype(float) / 255.0, target, p))
            else:
                db_noballs.append((bin_img.astype(float) / 255.0, target, p))

            # augment: gamma adjusted image
            # TODO do the augmentation after the image db creation in a extra script
            avg_img_f = np.average(img_normalized)
            if 0.2 <= avg_img_f <= 0.8:
                # augment: gamma
                for g in (0.4, 1.3):
                    if is_ball:
                        db_balls.append((adjust_gamma(img, g).astype(float) / 255.0, target, p))
                    else:
                        db_noballs.append((adjust_gamma(img, g).astype(float) / 255.0, target, p))

        print("len db_balls:", len(db_balls))
    return db_balls, db_noballs


def create_natural_segmentation_dataset(path, res):
    print("Loading images from " + path + " ...")
    db_balls = []
    db_noballs = []
    # parse csv file
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            f = os.path.join(os.path.dirname(path), row["filename"])
            p = row["filename"]

            # load image
            try:
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (res["x"], res["y"]))
                img_normalized = img.astype(float) / 255.0

            except Exception as ex:
                print("Error loading image ", f)
                continue

            is_ball = False
            # load ball information
            region_count = int(row["region_count"])
            if region_count > 0:
                atts = json.loads(row["region_attributes"])
                if atts["type"] == "smudged_ball":
                    # ignore this image
                    continue
                elif atts["type"] == "ball":
                    shape = json.loads(row["region_shape_attributes"])
                    if shape["name"] == "circle":
                        x_coord = int(shape["cx"])
                        y_coord = int(shape["cy"])
                        radius = int(shape["r"])

                        # draw detected circle into debug image
                        # cv2.circle(debug_img, (int(x),int(y)), int(radius), color=(0,0,255))
                        mask = np.zeros_like(img)
                        mask = cv2.circle(mask, (int(x_coord), int(y_coord)), int(radius), (255, 255, 255), -1)
                        mask_normalized = mask.astype(float) / 255.0

                        # normalize to resolution
                        x_coord = (x_coord / res["x"])
                        y_coord = (y_coord / res["y"])
                        radius = radius / max(res["x"], res["y"])

                        is_ball = True
                    else:
                        # we only support circles
                        print("WARNING: Annotation is not a circle")
                        continue
                elif atts["type"] == "smudge":
                    continue
                else:
                    # unknown type
                    print("Unknown type \"" + atts["type"] + "\" in file " + f)
                    continue
            else:
                # no region means no ball
                mask = np.zeros_like(img)
                mask_normalized = mask / 255.0

            # for each row add the image and the prediction
            mask_normalized = mask_normalized.reshape(*mask_normalized.shape, 1)
            if is_ball:
                target = mask_normalized
                db_balls.append((img_normalized, target, p))
            else:
                target = mask_normalized
                db_noballs.append((img_normalized, target, p))

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
