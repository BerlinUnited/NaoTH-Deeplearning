from cvat_sdk import make_client, models
from cvat_sdk.core.proxies.tasks import ResourceType, Task

with make_client(host="https://ball.informatik.hu-berlin.de/", credentials=('stella', 'Simon9247')) as client:
    task = client.tasks.retrieve(591)
    #print(tasks)
    task.export_dataset(format_name="YOLO 1.1", filename="./test.zip")