"""
    This script contains functions that download and prepare datasets for specific tasks
"""
import requests
from pathlib import Path
import zipfile

from cvat_tools import get_all_tasks, download_dataset, get_annotation_formats
from cvat_common_tools import login
from dataset_filter import get_coco_dataset_from_path, remove_unlabeled_images, save_coco_dataset, \
    create_ball_only_dataset, create_nao_only_dataset


def download_gopro_auto_annotation_data_rc19():
    """
        download gopro data from cvat.
        We first don't care if it's fully annotated. We simply remove every frame that has not a ball annotation in it
    """
    # Download part
    relevant_project_ids = [12, 13]  # the project ids that contain the rc19 gopro videos
    downloaded_datasets = list()

    with requests.Session() as session:
        login(session)
        exporter_format = get_annotation_formats(session)[0]  # coco format
        for project_id in relevant_project_ids:
            task_ids = get_all_tasks(session, project_id)
            for task_id in task_ids:
                # will download zipped datasets
                dataset_path = download_dataset(session, task_id, data_subfolder="combined",
                                                exporter_format=exporter_format)
                downloaded_datasets.append(dataset_path)

    return downloaded_datasets


def unzip_and_filter(downloaded_datasets):
    for zip_file in sorted(downloaded_datasets):
        print(zip_file)
        output_folder_name = Path(zip_file).with_suffix("")  # ../../108.zip becomes ../../108/

        # TODO catch errors here and remove folder if error occurs. Make sure ctrl+c is catched as well here
        with zipfile.ZipFile(str(zip_file), 'r') as zip_ref:
            zip_ref.extractall(str(output_folder_name))

        # filter part
        dataset = get_coco_dataset_from_path(output_folder_name)
        dataset = remove_unlabeled_images(dataset)

        ball_dataset = create_ball_only_dataset(dataset)
        nao_dataset = create_nao_only_dataset(dataset)
        save_coco_dataset(dataset, str(output_folder_name) + "_wo-unlabelled")
        save_coco_dataset(ball_dataset, str(output_folder_name) + "_wo-unlabelled_ball")
        save_coco_dataset(nao_dataset, str(output_folder_name) + "_wo-unlabelled_nao")


def main():
    downloaded_datasets = download_gopro_auto_annotation_data_rc19()
    unzip_and_filter(downloaded_datasets)


if __name__ == '__main__':
    main()
