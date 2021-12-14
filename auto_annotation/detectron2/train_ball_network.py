"""
    Train models for auto annotation in cvat
    Use naoth_ball folder for training a model that can detect balls in fullsize camera images

"""
import os
from pathlib import Path

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances

from detectron2.utils.logger import setup_logger

# Setup detectron2 logger
setup_logger()


def main():
    # load all coco datasets inside the data directory
    data_folder = Path("data")
    dataset_lists = [f for f in data_folder.iterdir() if f.is_dir()]

    for dataset in dataset_lists:
        print(str(dataset).split("/")[-1])
        # quit()
        register_coco_instances(str(dataset).split("/")[-1], {}, str(dataset) + "/annotations/instances_default.json",
                                str(dataset) + "/images")

    dataset_lists2 = list()
    for dataset in dataset_lists:
        # dataset = str(dataset).split("/")[-1]
        dataset_lists2.append(str(dataset).split("/")[-1])
        # print(dataset)

    # TODO build more augmentations here

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = tuple(dataset_lists2)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/retinanet_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 2000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ball). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    # cfg.MODEL.DEVICE = 'cpu'

    # create output folder
    counter = 0
    folder_name = os.path.join(cfg.OUTPUT_DIR, "ball_model_{}")
    while os.path.exists(folder_name.format(counter)):
        counter += 1
    folder_name = folder_name.format(counter)
    cfg.OUTPUT_DIR = folder_name
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == '__main__':
    main()
