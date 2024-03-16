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
import ultralytics
from tqdm import tqdm

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
    if not output.exists():
        mclient.fget_object(bucket_name, filename, output)

def test_visualize(x,y,width,height):
    img = cv2.imread("test_dataset/images/train/0005950.png")
    cv2.rectangle(img, (int(x), int(y)), (int(x+width), int(y+ height)), (0, 0, 255), 1)
    cv2.imwrite("test.png", img) 
    pass
    

def get_annotations(task_output, filename, output_folder):
    output_folder.mkdir(parents=True, exist_ok=True)
    output = Path(output_folder) / Path(filename).with_suffix(".txt")
    if output.exists():
        return
    with open(str(output), "w") as f:
        for anno in task_output['annotations']:
            results = anno["result"]
            # print(anno)
            for result in results:
                try:
                    # x,y,width,height are all percentages within [0,100]
                    x, y, width, height = result["value"]["x"], result["value"]["y"], result["value"]["width"], result["value"]["height"]
                    img_width = result['original_width']
                    img_height = result['original_height']
                    actual_label = result["value"]["rectanglelabels"][0]
                    label_id = label_dict[actual_label]
                except Exception as error:
                    print(f"annotations_list:´\n {task_output}")
                    print()
                    print("An exception occurred:", type(error).__name__, "–", error)
                    quit()

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
         TODO: if postgres query return multiple projects this function should account for it somehow -> worry about duplicate filenames
    """
    Path(dataset_name).mkdir(parents=True, exist_ok=True)

    # TODO get a row from postgres
    # get the minio bucket and ls labelstudio project id from the postgres
    def MyFn(project):
        return project.id
    
    existing_projects = [a for a in ls.list_projects()]
    print(f"exporting projects")
    for project in tqdm(sorted(existing_projects, key=MyFn)):
        task_ids = get_labeled_images(project)
        for task in task_ids:
            task_output = project.get_task(task)
            image_file_name = task_output["storage_filename"]
            #print(task_output['annotations'])

            label_path = Path(dataset_name) / "labels" / project.title
            img_path = Path(dataset_name) / "images" / project.title

            download_from_minio(project, image_file_name, img_path)
            get_annotations(task_output, image_file_name, label_path)

    # create yaml file
    # TODO can we do this automatically? -> like also with excluding segmentations and so on
    data = dict(
        path = f'../datasets/{dataset_name}',
        train = 'autosplit_train.txt', 
        val = 'autosplit_val.txt',
        names = {
            label_dict["ball"]: "ball",
            label_dict["nao"]: "nao",
            label_dict["penalty_mark"]: "penalty_mark"
        }
    )

    with open(f'{dataset_name}.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)
        
if __name__ == "__main__":
    export_dataset("test_dataset")
    # FIXME importing ultralytics takes a long time
    ultralytics.data.utils.autosplit('test_dataset/images', weights=(0.9, 0.1, 0.0), annotated_only=False)