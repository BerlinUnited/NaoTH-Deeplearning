"""
    functions to convert a coco dataset with multiple classes to one that has only one class because
    training on only one class might be better for auto annotation models.

    Additionally implemented functions to remove non annotated data

    REFERENCES:
        - https://openvinotoolkit.github.io/datumaro/docs/user-manual/media_formats/
        - https://openvinotoolkit.github.io/datumaro/docs/formats/coco/
        - https://openvinotoolkit.github.io/datumaro/docs/developer_manual/
"""
from pathlib import Path
import toml
from datumaro.components.dataset import Dataset

from common_tools import get_data_root


def get_coco_dataset_from_path(dataset_path):
    dataset = Dataset.import_from(str(dataset_path), "coco")
    return dataset


def save_coco_dataset(dataset, output_path="new_dataset"):
    dataset.export(output_path, format='coco_instances', save_images=True)


def remove_unlabeled_images(dataset):
    dataset.select(lambda item: len(item.annotations) != 0)
    return dataset


def create_ball_only_dataset(dataset):
    dataset.transform('remap_labels',
                      {
                          'ball': 'ball',  # keep the label by remapping it to the same name
                      }, default='delete')  # remove everything else
    return dataset


def create_nao_only_dataset(dataset):
    dataset.transform('remap_labels',
                      {
                          'nao': 'nao',  # keep the label by remapping it to the same name
                      }, default='delete')  # remove everything else
    return dataset
