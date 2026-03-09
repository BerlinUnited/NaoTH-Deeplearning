from vaapi.client import Vaapi
from label_studio_sdk import LabelStudio
import requests
from pathlib import Path
import os

log_ids = [679, 678, 677, 676, 675]
camera="BOTTOM"

def convert_to_yolo(bbox_data, class_map):
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


v_client = Vaapi(
    base_url=os.environ.get("VAT_API_URL"),
    api_key=os.environ.get("VAT_API_TOKEN"),
)

client = LabelStudio(
    base_url="https://labelstudio-api.berlin-united.com",
    api_key=os.environ.get("LABELSTUDIO_API_KEY"),
)

for log_id in log_ids:
    image_obj_list = v_client.image.list(log=log_id, camera=camera, validated=True)
    for img_obj in image_obj_list:
        img_url = "https://logs.berlin-united.com/" + img_obj.image_url
        Path(f"data/{camera}/images/train").mkdir(exist_ok=True, parents=True)
        save_path = Path(f"data/{camera}/images/train") / Path(img_obj.image_url).name
        img_data = requests.get(img_url).content
        with open(save_path, 'wb') as handler:
            handler.write(img_data)
        print(f"Downloaded: {save_path}")

        
        task_id = img_obj.labelstudio_url.split('=')[-1]
        # TODO download bounding box annotations from labelstudio with labelstudio sdk
        try:
            # Fetch the task details from Label Studio
            task = client.tasks.get(id=task_id)
            
            # Annotations are stored in a list (usually the first one is the 'completed' one)
            annotations = task.annotations
            
            if annotations:
                # result contains the actual bounding boxes/labels
                bbox_data = annotations[0].get('result', [])
                
                mapping = {"Ball": 0}
                yolo_results = convert_to_yolo(bbox_data, mapping)
                print(f"Retrieved {len(bbox_data)} annotation results for Task {task_id}")
                
                Path(f"data/{camera}/labels/train").mkdir(exist_ok=True, parents=True)
                label_path = f"{Path(f"data/{camera}/labels/train") / Path(img_obj.image_url).name.rsplit('.', 1)[0]}.txt"
                with open(label_path, 'w') as f:
                    f.write("\n".join(yolo_results))
            else:
                print(f"No annotations found for Task {task_id}")
                
        except Exception as e:
            print(f"Failed to fetch Task {task_id}: {e}")