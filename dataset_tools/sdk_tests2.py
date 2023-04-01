from cvat_sdk import make_client


def get_projects():
    with make_client(host="https://ball.informatik.hu-berlin.de/", credentials=('stella', 'Simon9247')) as client:
        projects = client.projects.list()
        my_project = projects[0]
        print(my_project)


def download_task(task_id, export_format="YOLO 1.1"):
    with make_client(host="https://ball.informatik.hu-berlin.de/", credentials=('stella', 'Simon9247')) as client:
        task = client.tasks.retrieve(task_id)
        task.export_dataset(format_name=export_format, filename="./test.zip")


#download_task(591)
get_projects()