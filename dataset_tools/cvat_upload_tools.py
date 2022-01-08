"""
    test upload tasks with data
    NOTE: this does not work yet
        - test if its possible to upload original data
"""
import time
import requests

from common_tools import cvat_login
from cvat_tools import get_all_tasks, get_labels_from_tasks


def create_task(session, task_dict, project_id):
    url = "https://ball.informatik.hu-berlin.de/api/v1/tasks"
    # setting this token is important for some reason but only for post requests, it seems
    csrftoken = session.cookies['csrftoken']

    data = {
        "name": task_dict["task_name"],
        "project_id": project_id,
        "overlap": 0,
        "segment_size": 1000,  # this works
        'csrfmiddlewaretoken': csrftoken
    }

    try:
        response = session.post(url, data=data)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)

    response = response.json()
    print(response)

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
    url = 'https://ball.informatik.hu-berlin.de/api/v1/tasks/{}/data'.format(task_id)
    response = session.post(url, data=data)
    print(response.json())

    # check for completion of data before returning
    while True:
        url = 'https://ball.informatik.hu-berlin.de/api/v1/tasks/{}/status'.format(task_id)
        response = session.get(url)
        print(response.json())
        if response.json()["state"] == "Finished":
            break
        time.sleep(10)


def upload_annotation(task_id):
    """
    the coordinate system for the rectangles starts at the top left corner and has the format:
    "points": [
        x1  y1   x2    y2
        0, 0, 100, 1000
      ]
    """
    #
    with requests.Session() as session:
        cvat_login(session)
        label_dict = get_labels_from_tasks(session, task_id)

    shape_list = []
    point_list = [
        [0, 0, 100, 100],
        [100, 100, 200, 200],
        [200, 200, 300, 300]
    ]
    for rect in point_list:
        shape_dict = {
            "type": "rectangle",
            "occluded": False,
            "z_order": 0,
            "rotation": 0,
            "points": rect,
            "frame": 0,
            "label_id": 209,
            "group": 0,
            "source": "string",
            "attributes": []
        }
        shape_list.append(shape_dict)
    tracks_dict = {
        "id": 0,
        "frame": 0,
        "label_id": 209,
        "group": 0,
        "source": "string",
        "shapes": [
            {
                "type": "rectangle",
                "occluded": False,
                "z_order": 0,
                "rotation": 0,
                "points": [0, 0, 0, 0],
                "id": 0,
                "frame": 0,
                "outside": False,
                "attributes": []
            }
        ],
        "attributes": []
    }
    annotation_dict = {
        "version": 0,
        "tags": [],
        "shapes": shape_list,
        "tracks": [tracks_dict]
    }

    with requests.Session() as session:
        cvat_login(session)
        url = "https://ball.informatik.hu-berlin.de/api/v1/tasks/416/annotations?action=update"
        csrftoken = session.cookies['csrftoken']

        try:
            response = session.patch(url, headers={"x-csrftoken": csrftoken}, json=annotation_dict)
            response.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise SystemExit(err)


def create_dummy_task():
    """
        this is an example of how the tasks can be created automatically
    """
    task_list = [
        {
            "path": "/repl/Experiments/cvat_test_images",
            "task_name": "dummy_task"
        }
    ]
    with requests.Session() as session:
        cvat_login(session)
        for task in task_list:
            create_task(session, task, project_id=5)


def delete_all_dummy_tasks():
    """
        Deletes all tasks in the TestProjekt project.
    """
    with requests.Session() as session:
        cvat_login(session)
        dummy_tasks = get_all_tasks(session, 5)
        for task_id in dummy_tasks:
            url = f"https://ball.informatik.hu-berlin.de/api/v1/tasks/{task_id}"
            csrftoken = session.cookies['csrftoken']
            try:
                response = session.delete(url, headers={"x-csrftoken": csrftoken})
                response.raise_for_status()
            except requests.exceptions.HTTPError as err:
                raise SystemExit(err)


if __name__ == '__main__':
    # create_dummy_task()
    upload_annotation(416)
    # delete_all_dummy_tasks()
