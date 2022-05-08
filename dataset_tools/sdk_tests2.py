from cvat_sdk import make_client, models
from cvat_sdk.core.proxies.tasks import ResourceType, Task

with make_client(host="https://ball.informatik.hu-berlin.de/", credentials=('stella', 'Simon9247')) as client:
    tasks = client.tasks.list()
    #task = client.tasks.export_dataset()