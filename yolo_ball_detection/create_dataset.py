"""

"""
import sys
import os

helper_path = os.path.join(os.path.dirname(__file__), '../tools')
sys.path.append(helper_path)


#from minio import Minio
from pathlib import Path
import yaml
import cv2
#import ultralytics
#from tqdm import tqdm

from os import environ

from helper import label_dict, get_minio_client, get_postgres_cursor, get_labelstudio_client

# setup connection to external services
mclient = get_minio_client()
pg_cur = get_postgres_cursor()
ls = get_labelstudio_client()



def download_from_minio_and_scale(bucket_name, filename, output_folder):
    # TODO is there a way to download the bytes and use them directly? 
    output = Path(output_folder) / filename
    #if not output.exists():
    mclient.fget_object(bucket_name, filename, output)
    # scale factor 8, but we define the final size here so that it can run multiple times 
    image = cv2.imread(str(output))
    resized_down = cv2.resize(image, (80, 60), interpolation= cv2.INTER_LINEAR)
    cv2.imwrite(str(output), resized_down)

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
                    # only export ball annotation - we just don't care about other labels right now
                    if actual_label != "ball":
                        continue
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

def get_datasets_bottom():
    # FIXME: it would be much cooler if we would save the id of the labelstudio project (but that cant be restored, i guess -> check it and document the final design decision somewhere)
    select_statement = f"""
    SELECT log_path, bucket_bottom FROM robot_logs WHERE bucket_bottom IS NOT NULL 
    """
    pg_cur.execute(select_statement)
    rtn_val = pg_cur.fetchall()
    logs = [x for x in rtn_val]
    return logs

def get_project_from_name(project_name):
    """
    In our database the project name is the same as the bucket name. For interacting with the labelstudio API we need the project ID
    """
    project_list = ls.list_projects() # TODO speed it up by creating the list only once outside the loop
    for project in project_list:
        if project.title == project_name:
            return project
    
    print("ERROR: Labelstudio project does not exist")

if __name__ == "__main__":
    data = get_datasets_bottom()
    dataset_name = Path("datasets") / Path("ball_only_dataset_80-60")
    Path(dataset_name).mkdir(parents=True, exist_ok=True)

    for logpath, bucketname in data:
        print(bucketname)
        project = get_project_from_name(bucketname)
        # get list of tasks
        task_ids = project.get_labeled_tasks_ids()
        for task in task_ids:
            image_file_name = project.get_task(task)["storage_filename"]
            img_path = Path(dataset_name) / "images" / project.title
            label_path = Path(dataset_name) / "labels" / project.title

            download_from_minio_and_scale(bucketname, image_file_name, img_path)
            get_annotations(project.get_task(task), image_file_name, label_path)


        break
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
