import os
import requests
import json
from pathlib import Path
from vaapi.client import Vaapi
from label_studio_sdk import LabelStudio


log_ids = [679, 678, 677, 676, 675]
camera = "BOTTOM"

IMAGE_SAVE_DIR = Path(f"data/{camera}/images")
IMAGE_SAVE_DIR.mkdir(exist_ok=True, parents=True)

ANNO_SAVE_DIR = Path(f"data/{camera}/annotations")
ANNO_SAVE_DIR.mkdir(exist_ok=True, parents=True)

v_client = Vaapi(
    base_url=os.environ.get("VAT_API_URL"),
    api_key=os.environ.get("VAT_API_TOKEN"),
)

client = LabelStudio(
    base_url="https://labelstudio-api.berlin-united.com/",
    api_key=os.environ.get("LABELSTUDIO_API_KEY"),
)

for log_id in log_ids:
    
    image_obj_list = v_client.image.list(log=log_id, camera=camera, validated=True)
    
    for img_obj in image_obj_list:
        img_url = "https://logs.berlin-united.com/" + img_obj.image_url
        img_filename = Path(img_obj.image_url).name
        
        img_path = IMAGE_SAVE_DIR / img_filename
        anno_filename = f"{Path(img_filename).stem}.json"
        anno_path = ANNO_SAVE_DIR / anno_filename

        if img_path.exists() and anno_path.exists():
            print(f"Schon gedownloadet: {img_filename}")
            continue 
        
        if not img_path.exists():
            response = requests.get(img_url)
            if response.status_code == 200:
                with open(img_path, 'wb') as f:
                    f.write(response.content)
            else:
                print(f"Fehler beim Download des Bildes: {img_url}")
                continue 

        if not anno_path.exists():
            task_id = img_obj.labelstudio_url.split('=')[-1]
            try:
                task = client.tasks.get(id=task_id)
                annotations = task.annotations
                
                if annotations:
                    bbox_data = annotations[0].get('result', [])
                    with open(anno_path, 'w', encoding='utf-8') as f:
                        json.dump(bbox_data, f, ensure_ascii=False, indent=4)
                        
                    print(f"Frisch gelagert: {img_filename}")
                else:
                    print(f"Keine Annotationen für Task {task_id}")
                    
            except Exception as e:
                print(f"Fehler beim Verarbeiten von Task {task_id}: {e}")