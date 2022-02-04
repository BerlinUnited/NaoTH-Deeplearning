import os
import sys
import argparse
import csv
import cppyy
import PIL.Image
from typing import NamedTuple, Tuple, List
import numpy as np
import time
import ctypes
import xml.etree.ElementTree as ET
from cppyy import addressof, bind_object
import cppyy.ll
from cppyy_tools import *
from pathlib import Path



def get_frames_for_dir(d, transform_to_squares=False):
    file_names = os.listdir(d)
    file_names.sort()
    imageFiles = [os.path.join(d, f) for f in file_names if os.path.isfile(
        os.path.join(d, f)) and f.endswith(".png")]

    result = list()
    for file in imageFiles:
        # open image to get the metadata
        img = PIL.Image.open(file)
        bottom = img.info["CameraID"] == "1"
        # parse camera matrix using metadata in the PNG file
        cam_matrix_translation = (float(img.info["t_x"]), float(
            img.info["t_y"]), float(img.info["t_z"]))

        cam_matrix_rotation = np.array(
            [
                [float(img.info["r_11"]), float(
                    img.info["r_12"]), float(img.info["r_13"])],
                [float(img.info["r_21"]), float(
                    img.info["r_22"]), float(img.info["r_23"])],
                [float(img.info["r_31"]), float(
                    img.info["r_32"]), float(img.info["r_33"])]
            ])

        # add ground truth (all actual balls) to the frame
        balls = list()

        frame = Frame(file, bottom, balls, cam_matrix_translation,
                      cam_matrix_rotation)

        result.append(frame)

    return result


if __name__ == "__main__":
    evaluator = PatchExecutor()
    with cppyy.ll.signals_as_exception():
        heinrich = [
            "/mnt/d/RoboCup/repo/NaoTH-DeepLearning/data_cvat/RoboCup2019/finished/7/obj_train_data/rc19-experiment/INDOOR/GOALY_SET_top/"]
        stella = [
            "/home/stella/RoboCup/Repositories/naoth-deeplearning/data_cvat/RoboCup2019/finished/7/obj_train_data/rc19-experiment/INDOOR/GOALY_SET_top/"]
        evaluator.execute(stella)
