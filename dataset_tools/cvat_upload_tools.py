"""
    test upload tasks with data
    NOTE: this does not work yet
        - test if its possible to upload original data
"""
import time
import requests

from cvat_common_tools import login


def create_task(session, test_dict, project_id):
    url = "https://ball.informatik.hu-berlin.de/api/v1/tasks"
    # setting this token is important for some reason but only for post requests, it seems
    csrftoken = session.cookies['csrftoken']

    data = {
        "name": test_dict["name"],
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
        'server_files[0]': test_dict["path"]
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
        time.sleep(20)


def create_gore21_tasks():
    """
        this is an example of how the tasks can be created automatically
    """
    task_list = [
        {
            "path": "/repl/2021-05-06_GORE21/2021-05-08_11-00-00_Berlin_United_vs_R-ZWEI-KICKERS_half1/extracted/0_12_Nao0054_210508-0947/combined_bottom.zip",
            "name": "2021-05-08_11-00-00_Berlin_United_vs_R-ZWEI-KICKERS_half1_0_12_Nao_bottom"
        }
    ]
    with requests.Session() as session:
        login(session)
        for task in task_list:
            create_task(session, task, project_id=6)


if __name__ == '__main__':
    create_gore21_tasks()
