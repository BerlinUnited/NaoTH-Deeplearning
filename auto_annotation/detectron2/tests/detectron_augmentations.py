"""
    test how augmentations look like with using the detectron2 augmentations

    TODO: check how to use the augmentation class
    TODO: implement the custom mapper from https://www.kaggle.com/dhiiyaur/detectron-2-compare-models-augmentation
    TODO: implement the training part
    TODO create evaluation
"""
# import some common libraries
import os
from pathlib import Path

# import some common detectron2 utilities
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import transforms as T
from detectron2.data import DatasetCatalog
from detectron2.data import detection_utils as utils
import copy

register_coco_instances("my_dataset", {},
 "task_rc19experiments-outdoor5-bottom-2020_06_29_21_54_22-coco 1.0/annotations/instances_default.json",
  "task_rc19experiments-outdoor5-bottom-2020_06_29_21_54_22-coco 1.0/images")

# inspect the dataset
#print(DatasetCatalog.get("my_dataset"))
#print(len(DatasetCatalog.get("my_dataset")))

dataset_dict = DatasetCatalog.get("my_dataset")[0]
dataset_dict = copy.deepcopy(dataset_dict)
image = utils.read_image(dataset_dict["file_name"], format="BGR")

transform_list = [T.Resize((800,800)),
                      T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                      T.RandomFlip(prob=0.5, horizontal=True, vertical=False), 
                      ]
image, transforms = T.apply_transform_gens(transform_list, image)
import cv2
cv2.imwrite("test.png", image) 
print(type(image))