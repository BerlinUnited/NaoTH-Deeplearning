"""
    This script contains functions that use the libs functions to create various ressources.
    e.g. download and filter datasets, create cvat tasks for specific events etc.
"""
from cvat_tools import download_all_tasks_from_project, create_tasks_from_json
from dataset_filter import unzip_and_filter


def create_gopro_rc19_datasets():
    # the project ids that contain the rc19 gopro videos
    downloaded_datasets = download_all_tasks_from_project(relevant_project_ids=[12, 13])
    unzip_and_filter(downloaded_datasets)


def create_rc19_ball_auto_annotation_datasets():
    # the project ids that contain the rc19 log images
    downloaded_datasets = download_all_tasks_from_project(relevant_project_ids=[4])
    unzip_and_filter(downloaded_datasets)


def create_rc19_own_gopro_tasks():
    # use incomplete json file here because video files are not fixed yet for one game
    create_tasks_from_json("rc19_own_gopro_incomplete.json", project_id=13)


def create_rc19_others_gopro_tasks():
    # use incomplete json file here because video files are not fixed yet for one game
    create_tasks_from_json("rc19_others_gopro_complete.json", project_id=12)


if __name__ == '__main__':
    create_rc19_others_gopro_tasks()
