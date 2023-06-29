import time
from cvat_sdk.api_client import Configuration, ApiClient, exceptions
from pprint import pprint

# Set up an API client
# Read Configuration class docs for more info about parameters and authentication methods
configuration = Configuration(
    host = "https://ball.informatik.hu-berlin.de/",
    username = 'stella',
    password = 'Simon9247',
)

with ApiClient(configuration) as api_client:
    format = "YOLO 1.1" # str | Desired output format name You can get the list of supported formats at: /server/annotation/formats
    id = 591 # int | A unique integer value identifying this task.
    #x_organization = "X-Organization_example" # str |  (optional)
    action = "download" # str | Used to start downloading process after annotation file had been created (optional) if omitted the server will use the default value of "download"
    #cloud_storage_id = 3.14 # float | Storage id (optional)
    filename = "/home/stella/filename_example.zip" # str | Desired output file name (optional)
    location = "local" # str | Where need to save downloaded dataset (optional)
    #org = "org_example" # str | Organization unique slug (optional)
    #org_id = 1 # int | Organization identifier (optional)
    use_default_location = False # bool | Use the location that was configured in task to export annotations (optional) if omitted the server will use the default value of True

    try:
        (data, response) = api_client.tasks_api.retrieve_dataset(
            format,
            id,
            #x_organization=x_organization,
            action=action,
            filename=filename,
            location=location,
            #org=org,
            #org_id=org_id,
            use_default_location=use_default_location,
        )
        pprint(data)
    except exceptions.ApiException as e:
        print("Exception when calling TasksApi.tasks_retrieve_dataset: %s\n" % e)