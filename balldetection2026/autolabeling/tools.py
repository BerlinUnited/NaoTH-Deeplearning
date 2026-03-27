from label_studio_sdk import LabelStudio
from typing import List
from pathlib import Path
import requests
import random
import yaml
import json
import os

def create_dataset_json(log_ids, camera, v_client, l_client, output_name):
    dataset = list()
    for log_id in log_ids:
        image_obj_list = v_client.image.list(log=log_id, camera=camera, validated=True)
        for img_obj in image_obj_list:
            img_url = "https://logs.berlin-united.com/" + img_obj.image_url
            
            task_id = img_obj.labelstudio_url.split('=')[-1]
            # download bounding box annotations from labelstudio with labelstudio sdk
            try:
                # Fetch the task details from Label Studio
                task = l_client.tasks.get(id=task_id)
                
                # Annotations are stored in a list (usually the first one is the 'completed' one)
                annotations = task.annotations
                
                if annotations:
                    # result contains the actual bounding boxes/labels
                    bbox_data = annotations[0].get('result', [])
                    
                    mapping = {"Ball": 0}
                    yolo_results = convert_to_yolo(bbox_data, mapping)
                    print(f"Retrieved {len(bbox_data)} annotation results for Task {task_id}")
                    
                    data = {
                        "frame_number": img_obj.frame.frame_number, "url":img_url, "labelstudio_url":img_obj.labelstudio_url,"annotations": yolo_results,
                    }
                    dataset.append(data)

                else:
                    print(f"No annotations found for Task {task_id}")
                    
            except Exception as e:
                print(f"Failed to fetch Task {task_id}: {e}")
    with open(output_name, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4)


def create_local_yolo_ds(dataset_file:str,output_path="datasets/custom_data", split_ratio=0.8) -> None:
    """
    Converts JSON metadata with URLs into a local YOLO dataset.
    """
    with open(dataset_file) as json_data:
        dataset = json.load(json_data)

    # 1. Create directory structure
    for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(output_path, folder), exist_ok=True)

    # Shuffle for random split
    random.shuffle(dataset)
    split_idx = int(len(dataset) * split_ratio)

    for i, entry in enumerate(dataset):
        # Determine if this goes to train or val
        subset = 'train' if i < split_idx else 'val'
        print(entry)
        url = entry['url']
        annotations = entry['annotations']
        
        # Create a unique filename based on the URL or index
        filename = Path(url).stem # e.g., '0008837'
        img_ext = Path(url).suffix # e.g., '.png'
        
        try:
            # 2. Download the image
            img_response = requests.get(url, timeout=10)
            if img_response.status_code == 200:
                img_path = os.path.join(output_path, f"images/{subset}/{filename}{img_ext}")
                with open(img_path, 'wb') as f:
                    f.write(img_response.content)
                
                # 3. Create the label file
                label_path = os.path.join(output_path, f"labels/{subset}/{filename}.txt")
                with open(label_path, 'w') as f:
                    for ann in annotations:
                        f.write(f"{ann}\n")
            else:
                print(f"Failed to download: {url}")
        except Exception as e:
            print(f"Error processing {url}: {e}")

    # TODO add creation of yaml file
    data = {
        'path': f'./{output_path}',  # dataset root dir
        'train': 'images/train',           # train images (relative to 'path')
        'val': 'images/val',               # val images (relative to 'path')
        'names': {
            0: 'ball'
        }
    }
    with open("dataset.yaml", 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"Dataset created at {output_path}")

def convert_to_yolo(bbox_data, class_map) -> List:
    """
    Converts Label Studio rectanglelabels to YOLO format strings.
    """
    yolo_lines = []
    
    for item in bbox_data:
        val = item['value']
        
        # 1. Get the class ID from the label list
        label_name = val['rectanglelabels'][0]
        class_id = class_map.get(label_name, 0) # Defaults to 0 if not found
        
        # 2. Extract Label Studio percentages (top-left)
        x_tl = val['x']
        y_tl = val['y']
        w_ls = val['width']
        h_ls = val['height']
        
        # 3. Calculate YOLO center coordinates (0.0 to 1.0)
        x_center = (x_tl + (w_ls / 2)) / 100
        y_center = (y_tl + (h_ls / 2)) / 100
        w_yolo = w_ls / 100
        h_yolo = h_ls / 100
        
        # 4. Format string: class x_center y_center width height
        # Using 6 decimal places for precision
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_yolo:.6f} {h_yolo:.6f}"
        yolo_lines.append(line)
        
    return yolo_lines

def predict_on_image(ls_client, task_id):


    prediction = {
        "task": task_id,
        "score": 0.95,
        "result": [{
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "score": 0.95, # Per-region score
            "value": {
                "x": 10, "y": 10, "width": 50, "height": 50,
                "rectanglelabels": ["Car"]
            }
        }]
    }

    ls_client.predictions.create(**prediction)



if __name__ == "__main__":
    client = LabelStudio(
        base_url="https://labelstudio-api.berlin-united.com",
        api_key=os.environ.get("LABELSTUDIO_API_KEY"),
    )
    predict_on_image(client, task_id=7825821)