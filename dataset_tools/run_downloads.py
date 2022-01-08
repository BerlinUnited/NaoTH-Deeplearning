"""
    This script contains functions that download and prepare datasets for specific tasks
"""
from pathlib import Path
import zipfile

from cvat_tools import download_all_tasks_from_project
from dataset_filter import get_coco_dataset_from_path, remove_unlabeled_images, save_coco_dataset, \
    create_ball_only_dataset, create_nao_only_dataset


def create_gopro_rc19_datasets():
    # the project ids that contain the rc19 gopro videos
    downloaded_datasets = download_all_tasks_from_project(relevant_project_ids=[12, 13])
    unzip_and_filter(downloaded_datasets)


def create_rc19_ball_auto_annotation_datasets():
    # the project ids that contain the rc19 log images
    downloaded_datasets = download_all_tasks_from_project(relevant_project_ids=[4])
    unzip_and_filter(downloaded_datasets)


def unzip_and_filter(downloaded_datasets):
    for zip_file in sorted(downloaded_datasets):
        print(zip_file)
        output_folder_name = Path(zip_file).with_suffix("")  # ../../108.zip becomes ../../108/

        with zipfile.ZipFile(str(zip_file), 'r') as zip_ref:
            zip_ref.extractall(str(output_folder_name))

        # filter part
        ball_dataset = get_coco_dataset_from_path(output_folder_name)
        ball_dataset = create_ball_only_dataset(ball_dataset)
        ball_dataset = remove_unlabeled_images(ball_dataset)

        nao_dataset = get_coco_dataset_from_path(output_folder_name)
        nao_dataset = create_nao_only_dataset(nao_dataset)
        nao_dataset = remove_unlabeled_images(nao_dataset)

        save_coco_dataset(ball_dataset, str(output_folder_name) + "_wo-unlabelled_ball")
        save_coco_dataset(nao_dataset, str(output_folder_name) + "_wo-unlabelled_nao")


def main():
    create_rc19_ball_auto_annotation_datasets()


if __name__ == '__main__':
    main()
