"""
    list of helpful functions to interact with our dataset in kaggle, university server and cvat
"""

import requests
import tqdm 
import json
from time import sleep
from pathlib import Path


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


def download_dataset(session, task_id):
    # TODO make url more generic
    url = 'https://ball.informatik.hu-berlin.de/api/v1/tasks/{}/dataset?format=YOLO%201.1&action=download'.format(task_id)
    local_filename = Path(__file__).absolute().parent.parent /'data/{}.zip'.format(task_id)
    # TODO check if original data is actually downloaded. This might be important in some cases

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


def download_unfinished_tasks(session):
    task_list = get_unfinished_tasks(session, project_id=2)
    for task_id in task_list:
        download_dataset(session, task_id)


with requests.Session() as session:
    login(session)
    #get_projects(session)
    #download_dataset(session, 66)
    download_unfinished_tasks(session)
    


    

