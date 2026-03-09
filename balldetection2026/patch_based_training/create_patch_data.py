import os
import requests
import cv2
import numpy as np
from pathlib import Path
from vaapi.client import Vaapi
from label_studio_sdk import LabelStudio

# --- CONFIGURATION ---
log_ids = [679, 678, 677, 676, 675]
camera = "TOP"
PATCH_SAVE_DIR = Path(f"data/{camera}/patches")
PATCH_SAVE_DIR.mkdir(exist_ok=True, parents=True)

# --- CLIENT SETUP ---
v_client = Vaapi(
    base_url=os.environ.get("VAT_API_URL"),
    api_key=os.environ.get("VAT_API_TOKEN"),
)

client = LabelStudio(
    base_url="https://labelstudio-api.berlin-united.com",
    api_key=os.environ.get("LABELSTUDIO_API_KEY"),
)

def save_image_patches(image_bytes, bbox_data, base_filename):
    """
    Decodes image, crops patches based on bbox_data, and saves them.
    """
    # Convert bytes to OpenCV image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        print(f"Error: Could not decode image {base_filename}")
        return

    h_img, w_img = img.shape[:2]

    for idx, item in enumerate(bbox_data):
        val = item.get('value', {})
        if 'rectanglelabels' not in val:
            continue
            
        # Label Studio uses percentages (0-100)
        x = val['x'] * w_img / 100
        y = val['y'] * h_img / 100
        w = val['width'] * w_img / 100
        h = val['height'] * h_img / 100

        # Convert to integer pixel coordinates
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))

        # Ensure coordinates are within image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        # Crop and Save
        patch = img[y1:y2, x1:x2]
        
        if patch.size > 0:
            resized_patch = cv2.resize(patch, (32, 32), interpolation=cv2.INTER_AREA)

            label = val['rectanglelabels'][0].replace(" ", "_")
            patch_name = f"{base_filename.stem}_{label}_{idx}.jpg"
            patch_path = PATCH_SAVE_DIR / patch_name
            cv2.imwrite(str(patch_path), resized_patch)
            # print(f"Saved patch: {patch_path}")

# --- MAIN LOOP ---
for log_id in log_ids:
    image_obj_list = v_client.image.list(log=log_id, camera=camera, validated=True)
    
    for img_obj in image_obj_list:
        img_url = "https://logs.berlin-united.com/" + img_obj.image_url
        img_filename = Path(img_obj.image_url).name
        
        # 1. Download image into memory
        response = requests.get(img_url)
        if response.status_code != 200:
            print(f"Failed to download {img_url}")
            continue
        img_bytes = response.content

        # 2. Get Annotations
        task_id = img_obj.labelstudio_url.split('=')[-1]
        try:
            task = client.tasks.get(id=task_id)
            annotations = task.annotations
            
            if annotations:
                bbox_data = annotations[0].get('result', [])
                
                # 3. Process patches
                save_image_patches(img_bytes, bbox_data, Path(img_filename))
                print(f"Processed {len(bbox_data)} patches for Task {task_id}")
            else:
                print(f"No annotations for Task {task_id}")
                
        except Exception as e:
            print(f"Failed to process Task {task_id}: {e}")