"""
    i could move this code somewhere else and upload it to datasets.naoth.de and then use something like this:
    ultralytics.data.utils.check_det_dataset(dataset, autodownload=True)

    One idea is to have this code in naoth_datasets, it creates datasets based on certain postgres queries -> then makes it a yolo dataset and uploads it to datasets.naoth.de 
    this repo then uses the generated yaml files to train (how should the yaml files go to the train code?)

    # TODO remove output folder if exists
"""

from label_studio_sdk import Client
from minio import Minio
from pathlib import Path
import yaml
import cv2

# Define the URL where Label Studio is accessible and the API key for your user account
LABEL_STUDIO_URL = 'https://ls.berlinunited-cloud.de/'
API_KEY = '6cb437fb6daf7deb1694670a6f00120112535687'

ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
ls.check_connection()

mclient = Minio("minio.berlinunited-cloud.de",
    access_key="naoth",
    secret_key="HAkPYLnAvydQA",
)

label_dict = {
    "ball": 0,
    "nao": 1,
    "penalty_mark": 2
}

def get_labeled_images(project):
    return project.get_labeled_tasks_ids()


def download_from_minio(project, filename, output_folder):
    bucket_name = project.title
    output = Path(output_folder) / filename
    mclient.fget_object(bucket_name, filename, output)

def test_visualize(x,y,width,height):
    img = cv2.imread("test_dataset/images/train/0005950.png")
    cv2.rectangle(img, (int(x), int(y)), (int(x+width), int(y+ height)), (0, 0, 255), 1)
    cv2.imwrite("test.png", img) 
    pass
    

def get_annotations(annotations_list, filename, output_folder):
    output = Path(output_folder) / Path(filename).with_suffix(".txt")
    with open(str(output), "w") as f:
        for anno in annotations_list:
            results = anno["result"]
            for result in results:
                # x,y,width,height are all percentages within [0,100]
                x, y, width, height, rotation = result["value"]["x"], result["value"]["y"], result["value"]["width"], result["value"]["height"], result["value"]["rotation"]
                img_width = result['original_width']
                img_height = result['original_height']
                actual_label = result["value"]["rectanglelabels"][0]
                label_id = label_dict[actual_label]
                print(result)

                # calculate the pixel coordinates -> visualization need it
                x_px = x / 100 * img_width
                y_px = y / 100 * img_height
                width_px = width / 100 * img_width
                height_px = height / 100 * img_height

                #calculate the center of the box
                cx = x_px + width_px / 2
                cy = y_px + height_px / 2

                # calculate the percentage in range [0,1]
                width = width / 100
                height = height / 100
                cx = cx / img_width
                cy = cy / img_height

                print(label_id, cx, cy, width, height)
                # format https://roboflow.com/formats/yolov5-pytorch-txt?ref=ultralytics
                f.write(f"{label_id} {cx} {cy} {width} {height}\n")

                # FIXME -> make me more general
                #test_visualize(x_px,y_px,width_px,height_px)

def export_dataset(dataset_name=""):
    """
         TODO: if postquery return multiple projects this function should account for it somehow -> worry about duplicate filenames
    """
    Path(dataset_name).mkdir(parents=True, exist_ok=True)
    train_img_path = Path(dataset_name) / "images" / "train"
    train_label_path = Path(dataset_name) / "labels" / "train"
    val_img_path = Path(dataset_name) / "images" / "val"
    val_label_path = Path(dataset_name) / "labels" / "val"
    train_img_path.mkdir(parents=True, exist_ok=True)
    train_label_path.mkdir(parents=True, exist_ok=True)
    val_img_path.mkdir(parents=True, exist_ok=True)
    val_label_path.mkdir(parents=True, exist_ok=True)

    # TODO get a row from postgres
    # get the minio bucket and ls labelstudio project id from the postgres
    existing_projects = [a for a in ls.list_projects()]
    my_project = existing_projects[1]

    task_ids = get_labeled_images(my_project)
    for task in task_ids:
        task_output = my_project.get_task(task)
        image_file_name = task_output["storage_filename"]
        #print(task_output['annotations'])
        download_from_minio(my_project, image_file_name, train_img_path)
        get_annotations(task_output['annotations'], image_file_name, train_label_path)

    # create yaml file
    # TODO find a good way to deal with the mapping between label id and name
    data = dict(
        path = f'../datasets/{dataset_name}',
        train = 'autosplit_train.txt', 
        val = 'autosplit_val.txt',
        names = {
            0: "ball",
            1: "nao",
            2: "penalty_mark"
        }
    )

    with open(f'{dataset_name}.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)
        

export_dataset("test_dataset")

import ultralytics
ultralytics.data.utils.autosplit('test_dataset/images', weights=(0.9, 0.1, 0.0), annotated_only=False)