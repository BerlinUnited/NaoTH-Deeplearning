"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
from PIL import Image
import cv2
import numpy as np
from config import *


class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x) for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            # in the loss function the predicted bounding boxes B are compared to the one groundtruth
            # TODO this probably means the labelmatrix can be shorter and does not have slots for multiple objects
            # TODO nao devils report that they predict a bounding box per class -> think about what needs to change for that
            if label_matrix[i, j, self.C] == 0:
                # Set that there exists an object
                label_matrix[i, j, self.C] = 1

                # Box coordinates
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                label_matrix[i, j, (self.C + 1) : (self.C + 5)] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix


class NaoTHDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        S=7,
        B=1,
        C=1,
        transform=None,
    ):
        img_height = 448
        img_width = 448
        self.S = S
        self.B = B
        self.C = C

        self.image_label_mapping = self.load_data_from_ultralytics_dataset()
        self.label_list = list()
        self.image_list = list()
        self.transform = transform

        for image_path, data in self.image_label_mapping.items():
            image = data["image"]
            label = data["label"]
            try:
                cls_id, cx, cy, w, h = label.split(" ")  # FIXME work with multiple labels per image
            except:
                # print("label", label)
                print("code cant deal with multiple annotations per image yet")
                continue

            self.label_list.append(label)
            self.image_list.append(image)

    def load_data_from_ultralytics_dataset(self):
        # Expect that all txt files contain labels
        # Path to the datasets
        dataset_dir = "datasets/ball_only_dataset_80-60"

        # Subdirectories for images and labels
        images_dir = os.path.join(dataset_dir, "images")
        labels_dir = os.path.join(dataset_dir, "labels")

        # Dictionary to store the mapping of images to their corresponding labels
        image_label_mapping = {}

        # Iterate over the subfolders in the images directory
        for folder_name in os.listdir(images_dir):
            images_subfolder_path = os.path.join(images_dir, folder_name)
            labels_subfolder_path = os.path.join(labels_dir, folder_name)

            # Ensure the corresponding labels subfolder exists
            if os.path.isdir(labels_subfolder_path):
                # Iterate over the image files in the current images subfolder
                for image_file in os.listdir(images_subfolder_path):
                    image_file_path = os.path.join(images_subfolder_path, image_file)

                    # Create the expected label file path
                    label_file_name = os.path.splitext(image_file)[0] + ".txt"
                    label_file_path = os.path.join(labels_subfolder_path, label_file_name)

                    # Ensure the label file exists
                    if os.path.isfile(label_file_path):
                        # Load the image
                        # image = Image.open(image_file_path)
                        # image = cv2.imread(str(image_file_path))
                        # image = image / 255.0
                        # resized = cv2.resize(image, (448,448), interpolation= cv2.INTER_LINEAR)
                        image = Image.open(str(image_file_path))
                        image = image.resize((448, 448))
                        # Read the label data
                        with open(label_file_path, "r") as f:
                            label_data = f.read()

                        # Store the image and its corresponding label in the mapping
                        # TODO I dont think that works well with multiple labels per image
                        image_label_mapping[image_file_path] = {
                            "image": image,
                            "label": label_data,
                        }
        return image_label_mapping

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # return self.image_list[index], self.label_list[index]
        boxes = []
        try:
            class_label, x, y, width, height = self.label_list[index].split(" ")
            class_label = int(class_label)
            x = float(x)
            y = float(y)
            width = float(width)
            height = float(height)
        except:
            print(self.label_list[index])
            quit()
        boxes.append([class_label, x, y, width, height])
        # img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = self.image_list[index]
        boxes = torch.tensor(boxes)

        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, self.C] == 0:
                # Set that there exists an object
                label_matrix[i, j, self.C] = 1

                # Box coordinates
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                label_matrix[i, j, self.C + 1 : self.C + 5] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix
