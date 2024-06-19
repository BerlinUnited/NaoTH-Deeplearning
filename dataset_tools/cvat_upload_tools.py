"""
    test upload tasks with data
    NOTE: - test if it is possible to upload to Organizations
        - TODO check how things look for other users
        - TODO: fix annotation upload function
"""

import requests
from common_tools.main import cvat_login
from cvat_tools import get_labels_from_tasks, create_task


def create_shapelist_from_points(point_list, label_id=209):
    # TODO handle multiple annotation in multiple images: point list might be unsufficent here
    shape_list = []

    for rect in point_list:
        shape_dict = {
            "type": "rectangle",
            "occluded": False,
            "z_order": 0,
            "rotation": 0,
            "points": rect,
            "frame": 0,
            "label_id": label_id,
            "group": 0,
            "source": "string",
            "attributes": [],
        }
        shape_list.append(shape_dict)

    return shape_list


def upload_track_annotation(task_id):
    """ """
    with requests.Session() as session:
        cvat_login(session)

    tracks_dict1 = {
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
                "attributes": [],
            },
            {
                "type": "rectangle",
                "occluded": False,
                "z_order": 0,
                "rotation": 0,
                "points": [200, 200, 300, 300],
                "id": 1,
                "frame": 4,
                "outside": False,
                "attributes": [],
            },
            # Set the outside property to end a track.
            # TODO how does it look like in exported formats
            {
                "type": "rectangle",
                "occluded": False,
                "z_order": 0,
                "rotation": 0,
                "points": [200, 200, 300, 300],
                "id": 2,
                "frame": 5,
                "outside": True,
                "attributes": [],
            },
        ],
        "attributes": [],
    }

    annotation_dict = {"version": 0, "tags": [], "shapes": [], "tracks": [tracks_dict1]}

    with requests.Session() as session:
        cvat_login(session)
        url = f"https://ball.informatik.hu-berlin.de/api/v1/tasks/{task_id}/annotations?action=update"
        csrftoken = session.cookies["csrftoken"]
        token = session.cookies["token"]
        try:
            response = session.patch(
                url, headers={"x-csrftoken": csrftoken, "Authorization": "token " + token}, json=annotation_dict
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise SystemExit(err)


def upload_simple_annotation(task_id):
    """
    This function sets the annotations for each frame without the use of the track feature and interpolation

    the coordinate system for the rectangles starts at the top left corner and has the format:
    "points": [
        x1  y1   x2    y2
        0, 0, 100, 1000
      ]
    # TODO test uploading global tags
    """
    #
    with requests.Session() as session:
        cvat_login(session)
        label_dict = get_labels_from_tasks(session, task_id)  # FIXME this must be used somewhere
        print(label_dict)

    point_list = [[0, 0, 100, 100], [100, 100, 200, 200], [200, 200, 300, 300]]
    shape_list = create_shapelist_from_points(point_list)

    annotation_dict = {"version": 0, "tags": [], "shapes": shape_list, "tracks": []}

    with requests.Session() as session:
        cvat_login(session)
        url = f"https://ball.informatik.hu-berlin.de/api/v1/tasks/{task_id}/annotations?action=update"
        csrftoken = session.cookies["csrftoken"]
        token = session.cookies["token"]
        try:
            response = session.patch(
                url, headers={"x-csrftoken": csrftoken, "Authorization": "token " + token}, json=annotation_dict
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise SystemExit(err)


def create_dummy_task_1():
    """
    this is an example of how the tasks can be created automatically

    """
    task_list = [{"path": "/repl/Experiments/cvat_test_images.zip", "name": "dummy_task_1"}]
    with requests.Session() as session:
        cvat_login(session)
        for task in task_list:
            create_task(session, task, project_id=5)


def create_dummy_task_2():
    """
    this is an example of how the tasks can be created automatically with track annotations
    """
    task_list = [{"path": "/repl/Experiments/tracking_test_dataset.zip", "name": "dummy_task_2"}]
    with requests.Session() as session:
        cvat_login(session)
        for task in task_list:
            create_task(session, task, project_id=5)


if __name__ == "__main__":
    # create_dummy_task_1()
    # create_dummy_task_2()
    # upload_simple_annotation(585)
    upload_track_annotation(585)
    pass
