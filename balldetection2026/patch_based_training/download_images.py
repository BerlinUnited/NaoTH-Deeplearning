import os
import requests
import argparse
import json
import sys
from pathlib import Path
from vaapi.client import Vaapi
from label_studio_sdk import LabelStudio


log_ids = [679, 678, 677, 676, 675, 683]

v_client = Vaapi(
    base_url=os.environ.get("VAT_API_URL"),
    api_key=os.environ.get("VAT_API_TOKEN"),
)

client = LabelStudio(
    base_url="https://labelstudio-api.berlin-united.com/",
    api_key=os.environ.get("LABELSTUDIO_API_KEY"),
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", type=str, help="Set BOTTOM or TOP")
    args = parser.parse_args()

    if args.camera is None:
        print("The camera is not set.\nSet with option -c, --camera TOP/BOTTOM")
        sys.exit()

    image_save_dir = Path(f"data/{args.camera}/images/all")
    image_save_dir.mkdir(exist_ok=True, parents=True)

    anno_save_dir = Path(f"data/{args.camera}/annotations/all")
    anno_save_dir.mkdir(exist_ok=True, parents=True)

    for log_id in log_ids:
        
        image_obj_list = v_client.image.list(log=log_id, camera=args.camera, validated=True)
        
        for img_obj in image_obj_list:
            img_url = "https://logs.berlin-united.com/" + img_obj.image_url
            img_filename = str(log_id) + "_" + Path(img_obj.image_url).name
            
            img_path = image_save_dir / img_filename
            anno_filename = f"{Path(img_filename).stem}.json"
            anno_path = anno_save_dir / anno_filename

            if img_path.exists() and anno_path.exists():
                print(f"Bereits gedownloadet: {img_filename}")
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
                            
                        print(f"Downloading: {img_filename}")
                    else:
                        print(f"Keine Annotationen für Task {task_id}")
                        
                except Exception as e:
                    print(f"Fehler bei {task_id}: {e}")