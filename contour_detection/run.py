from ultralytics import YOLO
from label_studio_sdk import LabelStudio
from pathlib import Path
import os
import cv2
import numpy as np

ls = LabelStudio(
    base_url="https://labelstudio-api.berlin-united.com",
    api_key=os.environ.get("LABELSTUDIO_API_KEY"),
)
model = YOLO('best.pt')

# Run inference
results = model("https://logs.berlin-united.com/2026-03-10-GO26/2026-03-11_11-50-00_Bit-Bots_vs_Berlin United_half2/extracted/3_35_Nao0022_260311-1200/log_bottom_jpg/0048080.png", conf=0.4, save=True)
result = results[0]
# Show the results (polygons drawn on image)
result.show()
print(result.boxes.conf)


predictions = list()

for i, mask in enumerate(result.masks.xyn): # .xyn gives normalized coordinates (0-1)
    
    raw_points = mask.copy() 
    
    # 2. Apply Approximation (Smoothing)
    # epsilon is the accuracy parameter. 
    # A smaller value (e.g., 0.001) keeps more detail.
    # A larger value (e.g., 0.01) makes it much smoother/simpler.
    epsilon = 0.005 * cv2.arcLength(raw_points.astype(np.float32), True)
    approx_points = cv2.approxPolyDP(raw_points.astype(np.float32), epsilon, True)
    
    # 3. Flatten and Convert to Label Studio 0-100 scale
    # approx_points comes back as (N, 1, 2) from OpenCV
    smoothed_points = []
    for p in approx_points:
        x, y = p[0]
        smoothed_points.append([float(x * 100), float(y * 100)])


    predictions.append({
        "from_name": "label",
        "to_name": "image",
        "type": "polygonlabels",
        "value": {
            "points": smoothed_points,
            "polygonlabels": ["Own Contour"]
        },
        "score": float(result.boxes.conf[i])
    })
    break

ls.predictions.create(
    task=7630726,
    score=0.6,
    result=predictions
)