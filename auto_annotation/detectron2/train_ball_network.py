"""
    Train models for auto annotation in cvat
    Use naoth_ball folder for training a model that can detect balls in fullsize camera images

"""
import os
from pathlib import Path
import toml
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances

from detectron2.utils.logger import setup_logger

# Setup detectron2 logger
setup_logger()


def get_data_root():
    with open('../../config.toml', 'r') as f:
        config_dict = toml.load(f)

    return config_dict["data_root"]


def get_rc19_ball_data():
    data_root = get_data_root()
    rc19_root_path = Path(data_root) / "data_cvat/RoboCup2019/combined/COCO_1.0"
    dataset_lists = []
    for path in rc19_root_path.iterdir():
        if path.is_dir() and str(path).endswith("_wo-unlabelled_ball"):
            # check if there are annotations
            annotation_path = path / "annotations"
            if not os.listdir(str(annotation_path)) == []:
                dataset_lists.append(path)

    return dataset_lists


def get_rc19_nao_data():
    data_root = get_data_root()
    rc19_root_path = Path(data_root) / "data_cvat/RoboCup2019/combined/COCO_1.0"

    dataset_lists = []
    for path in rc19_root_path.iterdir():
        if path.is_dir() and str(path).endswith("_wo-unlabelled_nao"):
            # check if there are annotations
            annotation_path = path / "annotations"
            if not os.listdir(str(annotation_path)) == []:
                dataset_lists.append(path)

    return dataset_lists


def register_datasets(dataset_lists):
    for dataset in dataset_lists:
        register_coco_instances(str(dataset).split("/")[-1], {}, str(dataset) + "/annotations/instances_default.json",
                                str(dataset) + "/images/default")


def create_config(dataset_lists):
    dataset_lists2 = list()
    for dataset in dataset_lists:
        dataset_lists2.append(str(dataset).split("/")[-1])

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
    # only has one class (ball).
    # (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect
    # uses num_classes+1 here.
    # cfg.MODEL.DEVICE = 'cpu'  # TODO this can be infered

    # create output folder
    counter = 0
    folder_name = os.path.join(cfg.OUTPUT_DIR, "ball_model_{}")
    while os.path.exists(folder_name.format(counter)):
        counter += 1
    folder_name = folder_name.format(counter)
    cfg.OUTPUT_DIR = folder_name
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


def main():
    # load the data
    dataset_lists = get_rc19_ball_data()
    register_datasets(dataset_lists)

    # TODO build more augmentations here

    cfg = create_config(dataset_lists)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == '__main__':
    main()
