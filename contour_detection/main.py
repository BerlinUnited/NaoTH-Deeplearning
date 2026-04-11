
import os
import requests
from label_studio_sdk import LabelStudio
from pathlib import Path

ls = LabelStudio(
        base_url="https://labelstudio-api.berlin-united.com",
        api_key=os.environ.get("LABELSTUDIO_API_KEY"),
    )

# --- SETUP ---
PROJECT_ID = 7860
OUTPUT_DIR = 'yolo_labels'

# Define your class mapping (must match your YOLO dataset.yaml)
LABEL_MAP = {
    "Own Contour": 0,
}


# 2. Retrieve all tasks with annotations
all_tasks = ls.tasks.list(project=PROJECT_ID)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for task in all_tasks:
    # Only process tasks that have at least one annotation
    if not task.annotations:
        continue
        
    
    
    yolo_lines = []
    
    # Label Studio can have multiple annotations per task; we usually want the first/latest
    annotation = task.annotations[0]
    #print(task)
    for result in annotation.get('result', []):
        # We only care about polygonlabels for segmentation
        if result['type'] == 'polygonlabels':
            label_name = result['value']['polygonlabels'][0]
            class_id = LABEL_MAP.get(label_name)
            
            if class_id is None:
                print(f"Warning: Label '{label_name}' not in map. Skipping.")
                continue
            
            # Extract points: Label Studio provides them as 0-100 (percentage)
            # YOLO needs them as 0.0-1.0
            points = result['value']['points']
            normalized_points = []
            for p in points:
                normalized_points.append(f"{p[0] / 100.0:.6f}") # x
                normalized_points.append(f"{p[1] / 100.0:.6f}") # y
            
            # Create the YOLO line: "class x1 y1 x2 y2 ..."
            yolo_line = f"{class_id} {' '.join(normalized_points)}"
            yolo_lines.append(yolo_line)



    # 3. Save the annotations to a file
    if yolo_lines:
        # download image
        try:

            url = task.data["image"]
            # Get the image filename to name the .txt file
            filename = Path(url).stem  # e.g., '0008837'
            img_ext = Path(url).suffix  # e.g., '.png'
            #filename = os.path.splitext(os.path.basename(img_path))[0]
            #label_path = os.path.join(OUTPUT_DIR, f"{filename}.txt")
            # 2. Download the image
            img_response = requests.get(url, timeout=10)
            if img_response.status_code == 200:
                img_path = os.path.join(OUTPUT_DIR, f"images/train/{filename}{img_ext}")
                with open(img_path, "wb") as f:
                    f.write(img_response.content)

                # 3. Create the label file
                label_path = os.path.join(
                    OUTPUT_DIR, f"labels/train/{filename}.txt"
                )
                with open(label_path, "w") as f:
                    f.write('\n'.join(yolo_lines))
            else:
                print(f"Failed to download (Status {img_response.status_code}): {url}")
        except Exception as e:
            print(f"Error processing {url}: {e}")

print(f"Export complete. Labels saved in: {OUTPUT_DIR}")