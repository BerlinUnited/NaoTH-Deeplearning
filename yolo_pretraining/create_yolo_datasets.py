"""
    i could move this code somewhere else and upload it to datasets.naoth.de and then use something like this:
    ultralytics.data.utils.check_det_dataset(dataset, autodownload=True)

    One idea is to have this code in naoth_datasets, it creates datasets based on certain postgres queries -> then makes it a yolo dataset and uploads it to datasets.naoth.de 
    this repo then uses the generated yaml files to train (how should the yaml files go to the train code?)

    # TODO remove output folder if exists
"""
import zipfile
from label_studio_sdk import Client
from minio import Minio
from pathlib import Path
import yaml
from os import environ
import shutil
import ultralytics
import argparse
from tqdm import tqdm
import datetime

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

def download_from_minio(project, filename, output_folder):
    bucket_name = project.title
    output = Path(output_folder) / filename
    if not output.exists():
        mclient.fget_object(bucket_name, filename, output)

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

                #print(label_id, cx, cy, width, height)
                # format https://roboflow.com/formats/yolov5-pytorch-txt?ref=ultralytics
                f.write(f"{label_id} {cx} {cy} {width} {height}\n")

                # FIXME -> make me more general
                #test_visualize(x_px,y_px,width_px,height_px)

def get_projects_bottom():
    # 175 is partially broken TODO: how to account for that?
    project_id_list = [183, 182, 181, 180, 179, 178, 177, 176, 175, 174, 173, 172, 171, 170, 169, 
                       168, 167, 166, 165, 164, 159, 160, 161, 162, 163, 157, 156, 155, 154, 149,
                       150, 151, 152, 153, 148, 147, 146]
    projects= []
    for id in project_id_list:
        projects.append(ls.get_project(id))
    return projects

def get_projects_top():
    project_id_list = [108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123,
                       124, 125, 126, 127, 128, 129, 130, 131, 132, 138, 139, 140, 144, 145]
    projects= []
    for id in project_id_list:
        projects.append(ls.get_project(id))
    return projects

def export_dataset(dataset_name="", camera=""):
    """
    """
    Path(dataset_name).mkdir(parents=True, exist_ok=True)

    # TODO data from postgres that account for broken images
    # get the minio bucket and ls labelstudio project id from the postgres
    def my_sort_function(project):
        return project.id
    
    if camera == "bottom":
        existing_projects = get_projects_bottom()
    elif camera == "top":
        existing_projects = get_projects_top()
    else:
        print("ERROR: not a valid camera argument")
        quit()
    print(f"exporting projects")
    for project in tqdm(sorted(existing_projects, key=my_sort_function)):
        tasks = project.get_labeled_tasks()
        for task_output in tasks:
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

def zip_and_upload_datasets(dataset_name):
    filenames = [f"{dataset_name}.yaml"]
    directory = Path(dataset_name)

    with zipfile.ZipFile(f"{dataset_name}.zip", mode="w") as archive:
        for filename in filenames:
            archive.write(filename, arcname=Path(filename).relative_to(directory.parent))

        for file_path in directory.rglob("*"):
            archive.write(file_path, arcname=file_path.relative_to(directory.parent))

    remote_dataset_path = Path(environ.get("REPL_ROOT")) / "datasets"

    # FIXME: will overwrite - not good for debugging
    zip_file_name = Path(dataset_name.name).with_suffix(".zip")
    output_file_path = remote_dataset_path / zip_file_name
    shutil.copyfile(f"{dataset_name}.zip", output_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", required=True, choices=['bottom', 'top'])
    args = parser.parse_args()

    now = datetime.datetime.now().strftime('%Y-%m-%d')
    dataset_name= Path("datasets") / f"yolo-full-size-detection_dataset_{args.camera}_{now}"
    export_dataset(dataset_name, args.camera)
    # FIXME importing ultralytics takes a long time - maybe use sklearn to split or write my own function
    ultralytics.data.utils.autosplit(f'{dataset_name}/images', weights=(0.9, 0.1, 0.0), annotated_only=False)
    
    # zip and upload created dataset
    zip_and_upload_datasets(dataset_name)