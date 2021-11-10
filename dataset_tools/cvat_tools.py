"""
    list of helpful functions to interact with our dataset in kaggle, university server and cvat
"""

import requests
import tqdm 
import json
from time import sleep
from pathlib import Path
import zipfile
import fileinput
import os

def login(session):
    # TODO document credential file
    with open('.credentials') as f:
        secrets = json.load(f)

    login_data = {'username':secrets["name"], 'password': secrets["pass"]}
    response = session.post('https://ball.informatik.hu-berlin.de/api/v1/auth/login', data=login_data)
    # TODO add error handling here


def get_projects(session):
    response = session.get('https://ball.informatik.hu-berlin.de/api/v1/projects', headers={'accept': 'application/json'})
    # TODO handle pagination and error handling

    result_list = response.json()["results"]
    for result in result_list:
        print(result["name"], result["id"])


def get_annotation_formats(session):
    response = session.get('https://ball.informatik.hu-berlin.de/api/v1/server/annotation/formats')

    print(response.json()['exporters'][0]['name'])


def download_dataset(session, task_id, data_subfolder):
    # TODO make url more generic regarding annotation format
    url = 'https://ball.informatik.hu-berlin.de/api/v1/tasks/{}/dataset?format=YOLO%201.1&action=download'.format(task_id)

    local_filename = Path(__file__).absolute().parent.parent /'data/{}/{}.zip'.format(data_subfolder, task_id)
    # TODO check if original data is actually downloaded. This might be important in some cases
    print(local_filename)
    # create folder
    Path(local_filename.parent).mkdir(parents=True, exist_ok=True)

    if Path(local_filename).exists():
        return

    with session.get(url, stream=True) as r:
        r.raise_for_status()
        while r.status_code == 202:
            r = session.get(url, stream=True)
            sleep(1)

        total_size_in_bytes= int(r.headers.get('content-length', 0))
        progress_bar = tqdm.tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                progress_bar.update(len(chunk))
                f.write(chunk)
        progress_bar.close()

    # TODO add unzip functionality here


def get_unfinished_tasks(session, project_id):
    url = "https://ball.informatik.hu-berlin.de/api/v1/projects/" + str(project_id)
    response = session.get(url)
    response_dict = response.json()

    unfinished_tasks = list()
    task_list = response_dict["tasks"]
    for task in task_list:
        if task["status"] == "completed":
            continue

        unfinished_tasks.append(task["id"])

    return unfinished_tasks


def get_finished_tasks(session, project_id):
    url = "https://ball.informatik.hu-berlin.de/api/v1/projects/" + str(project_id)
    response = session.get(url)
    response_dict = response.json()

    finished_tasks = list()
    task_list = response_dict["tasks"]
    for task in task_list:
        if task["status"] == "completed":
            finished_tasks.append(task["id"])        

    return finished_tasks


def download_unfinished_tasks(session):
    task_list = get_unfinished_tasks(session, project_id=2)
    for task_id in task_list:
        download_dataset(session, task_id, "unfinished")


def download_finished_tasks(session):
    task_list = get_finished_tasks(session, project_id=2)
    print(task_list)
    for task_id in task_list:
        download_dataset(session, task_id, "finished")


def unpack_yolo_zips():
    # FIXME don't unpack if folder exist already
    
    input_folder_name = Path(__file__).absolute().parent.parent /'data/finished'
    """
    zip_list = Path(input_folder_name).glob('*.zip')
    for zip_file in zip_list:
        print(zip_file)
        output_folder_name = input_folder_name / zip_file.stem
        with zipfile.ZipFile(str(zip_file), 'r') as zip_ref:
            zip_ref.extractall(str(output_folder_name))
    """
    fix_yolo_files(input_folder_name)
    # -------------
    """
    input_folder_name = Path(__file__).absolute().parent.parent /'data/unfinished'
    zip_list = Path(input_folder_name).glob('*.zip')
    for zip_file in zip_list:
        print(zip_file)
        output_folder_name = input_folder_name / zip_file.stem
        with zipfile.ZipFile(str(zip_file), 'r') as zip_ref:
            zip_ref.extractall(str(output_folder_name))

    fix_yolo_files(input_folder_name)
    """

def fix_yolo_files(folder_name):
    # modify the contents in the unzipped folders so it reflects the actual paths
    sub_folders = [f for f in folder_name.iterdir() if f.is_dir()]
    
    for folder in sorted(sub_folders):
        print(folder)
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


with requests.Session() as session:
    login(session)
    #get_projects(session)
    #download_dataset(session, 66)
    #download_unfinished_tasks(session)
    #download_finished_tasks(session)
    unpack_yolo_zips()

    # TODO add function to combine train.txt files



    

