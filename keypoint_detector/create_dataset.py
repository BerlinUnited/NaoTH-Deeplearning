"""

"""
import sys
import os

helper_path = os.path.join(os.path.dirname(__file__), '../tools')
sys.path.append(helper_path)

from pathlib import Path
import yaml
import cv2


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

def get_annotations(task_output, filename, label_path, img_path):
    label_path.mkdir(parents=True, exist_ok=True)
    txt_file = Path(label_path) / Path(filename).with_suffix(".txt")
    
    img_path.mkdir(parents=True, exist_ok=True)
    img_file = Path(img_path) / filename
    
    #if txt_file.exists():
    #    return
    
    # TODO not downloading images when there is no ball annotations
    # TODO ask chatgpt for a script that can use such a dataset for yolo
    ball_bbox_list = list()
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
                else:
                    label_id = label_dict[actual_label]
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

                    ball_bbox_list.append((label_id,cx, cy, width, height))

                    
            except Exception as error:
                print(f"annotations_list:´\n {task_output}")
                print()
                print("An exception occurred:", type(error).__name__, "–", error)
                quit()
        
    
    
    if len(ball_bbox_list) > 0:
        with open(str(txt_file), "w") as f:
            for bbox in ball_bbox_list:
                # format https://roboflow.com/formats/yolov5-pytorch-txt?ref=ultralytics
                #(label_id, cx, cy, width, height)
                print(bbox)
                f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")

        # download image
        bucket_name = label_path.name # HACK we already have the folder structure here and we named the folder according to the bucket/project name
        print(f"\tbucket_name: {bucket_name}")
        print(f"\tfilename: {filename}")
        print(f"\timg_file: {img_file}")
        mclient.fget_object(bucket_name, filename, img_file)
        # scale factor 8, but we define the final size here so that it can run multiple times 
        image = cv2.imread(str(img_file))
        resized_down = cv2.resize(image, (80, 60), interpolation= cv2.INTER_LINEAR)
        cv2.imwrite(str(img_file), resized_down)

    # cleanup empty folders
    if not any(label_path.iterdir()):
        os.rmdir(str(label_path))
    if not any(img_path.iterdir()):
        os.rmdir(str(img_path))


def get_datasets_bottom():
    select_statement = f"""        
    SELECT log_path, bucket_bottom, ls_project_bottom
    FROM robot_logs
    WHERE bucket_bottom IS NOT NULL AND ls_project_bottom IS NOT NULL
    AND bottom_validated = true
    """
    pg_cur.execute(select_statement)
    rtn_val = pg_cur.fetchall()
    logs = [x for x in rtn_val]
    return logs


if __name__ == "__main__":
    data = get_datasets_bottom()
    dataset_name = Path("datasets") / Path("ball_only_dataset_80-60")  # TODO add date
    Path(dataset_name).mkdir(parents=True, exist_ok=True)

    for logpath, bucketname, ls_project_bottom in data:
        print(bucketname)
        ls_project = ls.get_project(id=ls_project_bottom)

        # get list of tasks
        labeled_tasks = ls_project.get_labeled_tasks_ids()
        for task in labeled_tasks:
            image_file_name = ls_project.get_task(task)["storage_filename"]
            img_path = Path(dataset_name) / "images" / ls_project.title
            label_path = Path(dataset_name) / "labels" / ls_project.title

            #download_from_minio_and_scale(bucketname, image_file_name, img_path)
            get_annotations(ls_project.get_task(task), image_file_name, label_path, img_path)


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

    # TODO upload dataset