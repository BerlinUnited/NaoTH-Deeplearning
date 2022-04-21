"""
    list of helpful functions to interact with our dataset in kaggle, university server and cvat

    Downloads are done in the configured download folder with the following structure

    data_cvat
    ->Project Name
        -> unfinished
            -> Coco Format
            -> Yolo Format
        -> finished
            -> Coco Format
            -> Yolo Format
        -> combined
            -> Coco Format
            -> Yolo Format

    NOTE:
        - only yolo and coco formats are currently used be other scripts
"""

import requests
from tqdm import tqdm
from time import sleep
from pathlib import Path
import zipfile
import os
import json
import urllib.parse

from common_tools.main import cvat_login, get_data_root


def get_projects(session):
    try:
        response = session.get('https://ball.informatik.hu-berlin.de/api/v1/projects')
        response.raise_for_status()

        # TODO handle pagination
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)

    result_list = response.json()["results"]
    for result in result_list:
        print(result["name"], result["id"])


def get_annotation_formats(session):
    try:
        response = session.get('https://ball.informatik.hu-berlin.de/api/server/annotation/formats')
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)

    exporter_list = list()
    for exporter_format in response.json()['exporters']:
        exporter_list.append(exporter_format['name'])

    return exporter_list


def get_project_name(session, project_id):
    project_url = f"https://ball.informatik.hu-berlin.de/api/projects/{project_id}"
    try:
        response = session.get(project_url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)

    project_details = response.json()
    return project_details["name"]


def download_dataset(session, task_id, data_subfolder="unfinished", export_format="YOLO 1.1"):
    # get task details
    task_url = f'https://ball.informatik.hu-berlin.de/api/tasks/{task_id}'
    try:
        response = session.get(task_url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)

    task_details = response.json()

    # build the output folder structure
    project_name = get_project_name(session, task_details["project_id"]).replace(" ", "_")
    format_name = export_format.replace(" ", "_")

    local_filename = Path(get_data_root()) / f'data_cvat/{project_name}/{data_subfolder}/{format_name}/{task_id}.zip'
    Path(local_filename.parent).mkdir(parents=True, exist_ok=True)
    if Path(local_filename).exists():
        return local_filename
    format_quoted = urllib.parse.quote(export_format)
    url = f'https://ball.informatik.hu-berlin.de/api/tasks/{task_id}/dataset?format={format_quoted}&action=download'

    with session.get(url, stream=True) as r:
        r.raise_for_status()
        while r.status_code == 202:
            r = session.get(url, stream=True)
            sleep(1)

        total_size_in_bytes = int(r.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                progress_bar.update(len(chunk))
                f.write(chunk)
        progress_bar.close()

    return local_filename


def get_unfinished_tasks(session, project_id):
    url = f"https://ball.informatik.hu-berlin.de/api/projects/{project_id}"
    try:
        response = session.get(url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)

    response_dict = response.json()

    unfinished_tasks = list()
    task_list = response_dict["tasks"]
    for task in task_list:
        if task["status"] == "completed":
            continue

        unfinished_tasks.append(task["id"])

    return unfinished_tasks


def get_finished_tasks(session, project_id):
    url = f"https://ball.informatik.hu-berlin.de/api/projects/{project_id}"
    try:
        response = session.get(url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)

    response_dict = response.json()

    finished_tasks = list()
    task_list = response_dict["tasks"]
    for task in task_list:
        if task["status"] == "completed":
            finished_tasks.append(task["id"])

    return finished_tasks


def get_all_tasks(session, project_id):
    # TODO can be combine with finished and unfinished to one function
    url = f"https://ball.informatik.hu-berlin.de/api/projects/{project_id}"
    try:
        response = session.get(url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)

    response_dict = response.json()

    all_tasks = list()
    task_list = response_dict["tasks"]

    for task in task_list:
        all_tasks.append(task)

    return all_tasks


def download_unfinished_tasks(session, project_id, exporter_format):
    task_list = get_unfinished_tasks(session, project_id)
    for task_id in task_list:
        download_dataset(session, task_id, data_subfolder="unfinished", exporter_format=exporter_format)


def download_finished_tasks(session, project_id, exporter_format):
    task_list = get_finished_tasks(session, project_id)

    for task_id in task_list:
        download_dataset(session, task_id, data_subfolder="finished", exporter_format=exporter_format)


def download_all_tasks_from_project(relevant_project_ids, export_format=None):
    """
        returns a list of the paths to the downloaded zip files
    """
    downloaded_datasets = list()

    with requests.Session() as session:
        cvat_login(session)
        available_export_formats = get_annotation_formats(session)
        if export_format is None:
            export_format = 'COCO 1.0'

        elif export_format not in available_export_formats:
            print("ERROR: Only the following export formats are supported")
            print(available_export_formats)

        for project_id in relevant_project_ids:
            task_ids = get_all_tasks(session, project_id)
            for task_id in task_ids:
                # will download zipped datasets
                dataset_path = download_dataset(session, task_id, data_subfolder="combined",
                                                export_format=export_format)
                downloaded_datasets.append(dataset_path)

    return downloaded_datasets


def get_labels_from_tasks(session, task_id):
    """
        get the label names and id's that are valid for a given task

        returns a dict like
        {
            name: id,
            name: id
        }
        the label id's are needed for uploading annotations to a task
    """
    task_url = f'https://ball.informatik.hu-berlin.de/api/tasks/{task_id}'
    try:
        response = session.get(task_url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)

    labels_list = response.json()["labels"]
    label_dict = {}
    for label in labels_list:
        label_dict.update({label["name"]: label["id"]})

    return label_dict


def unpack_zips(downloaded_datasets):
    """
        
    """

    for zip_file in sorted(downloaded_datasets):
        print(zip_file)
        output_folder_name = zip_file.with_suffix("")  # ../../108.zip becomes ../../108/

        # TODO catch errors here and remove folder if error occurs. Make sure ctrl+c is catched as well here
        with zipfile.ZipFile(str(zip_file), 'r') as zip_ref:
            zip_ref.extractall(str(output_folder_name))


def fix_yolo_files(folder):
    """
        modify the contents in the unzipped folders so it reflects the actual paths
    """
    # change paths in obj.data
    obj_data_path = folder / "obj.data"
    train_data_path = folder / "obj_train_data"
    train_txt_path = folder / "train.txt"
    obj_name_path = folder / "obj.names"

    new_lines = list()
    with open(str(obj_data_path), "r") as f:
        lines = [line.strip() for line in f if line.strip()]
        for line in lines:
            if line.startswith("train"):
                new_line = "train = {}".format(train_txt_path)
            elif line.startswith("names"):
                new_line = "names = {}".format(obj_name_path)
            else:
                new_line = line
            new_lines.append(new_line)

    with open(str(obj_data_path), "w") as file1:
        file1.writelines(line + '\n' for line in new_lines)

    # change paths in train.txt
    f = open(str(train_txt_path), "w")

    # TODO add annotations to train.txt
    print(train_data_path)
    all_image_paths = list(Path(train_data_path).glob('**/*.png'))
    for path in sorted(all_image_paths):
        txt_path = path.with_suffix(".txt")
        annotation_data = str(path)

        if txt_path.exists() and os.path.getsize(str(txt_path)) > 0:
            g = open(str(txt_path), "r")
            lines = [line.strip() for line in g if line.strip()]
            for line in lines:
                # FIXME: order of values and scaling is wrong
                annotation_data += " " + str(line).replace(" ", ",")

        print(annotation_data)
        f.write(annotation_data + "\n")


def delete_all_tasks_in_project(project_id):
    """
        Deletes all tasks in the TestProjekt project.
    """
    with requests.Session() as session:
        cvat_login(session)
        tasks = get_all_tasks(session, project_id)

        for task_id in tqdm(tasks):
            url = f"https://ball.informatik.hu-berlin.de/api/tasks/{task_id}?org="
            csrftoken = session.cookies['csrftoken']
            token = session.cookies['token']
            try:
                response = session.delete(url, headers={"x-csrftoken": csrftoken, "Authorization": "token " + token})

                response.raise_for_status()
            except requests.exceptions.HTTPError as err:
                raise SystemExit(err)


def create_task(session, task_dict, project_id):
    url = "https://ball.informatik.hu-berlin.de/api/tasks?org="
    # setting this token is important for some reason but only for post requests, it seems
    csrftoken = session.cookies['csrftoken']
    token = session.cookies['token']
    data = {
        "name": task_dict["name"],
        "project_id": project_id,
        "overlap": 0,
        "segment_size": 1000,  # this works
        'csrfmiddlewaretoken': csrftoken
    }

    try:
        response = session.post(url, data=data, headers={"Authorization": "token " + token})
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)

    response = response.json()

    data = {
        'image_quality': 100,
        'csrfmiddlewaretoken': csrftoken,
        'compressed_chunk_type': "imageset",
        'storage': "local",
        'storage_method': "file_system",
        'copy_data': True,
        # use_cache': True  makes the task creation fast, viewing the frames is slower. It will load for a couple of
        # seconds every 200 frames
        'use_cache': True,
        'server_files[0]': task_dict["path"]
    }
    task_id = response['id']

    # add data to task
    url = f'https://ball.informatik.hu-berlin.de/api/tasks/{task_id}/data?org='
    response = session.post(url, data=data, headers={"Authorization": "token " + token})

    # check for completion of data before returning
    while True:
        url = 'https://ball.informatik.hu-berlin.de/api/tasks/{}/status'.format(task_id)
        response = session.get(url)
        print(response.json())
        if response.json()["state"] == "Finished":
            break
        elif response.json()["state"] == "Failed":
            break
        sleep(10)


def create_tasks_from_json(json_file, project_id):
    """
        this is an example of how the tasks can be created automatically
    """
    with open(json_file) as f:
        task_list = json.load(f)

    with requests.Session() as session:
        cvat_login(session)
        for task in task_list:
            create_task(session, task, project_id=project_id)


def main():
    with requests.Session() as session:
        cvat_login(session)
        output = get_annotation_formats(session)
        print(output)
        # get_projects(session)
        # exporters = get_annotation_formats(session)
        # exporters[0] is coco
        # exporters[11] is yolo

        # download_dataset(session, task_id=66, exporter_format=exporters[11])
        # download_dataset(session, task_id=66, exporter_format=exporters[0])
        # download_unfinished_tasks(session, project_id=3, exporter_format=exporters[11])
        # download_finished_tasks(session, project_id=3, exporter_format=exporters[11])
        #unpack_zips()

        # TODO add function to combine train.txt files
        # TODO the fix_yolo functions should go to another script


if __name__ == '__main__':
    main()
