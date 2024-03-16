"""
    run the best model on new data in labelstudio
"""
import sys
from pathlib import Path
from label_studio_sdk import Client
from minio import Minio
import psycopg2
import random
import string
import requests
from ultralytics import YOLO
from os import environ

LABEL_STUDIO_URL = "https://ls.berlinunited-cloud.de/"
API_KEY = "6cb437fb6daf7deb1694670a6f00120112535687"
ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
ls.check_connection()

mclient = Minio("minio.berlinunited-cloud.de",
    access_key="naoth",
    secret_key="HAkPYLnAvydQA",
)

def download_from_minio(project, filename, output_folder):
    bucket_name = project.title
    output = Path(output_folder) / filename
    mclient.fget_object(bucket_name, filename, output)

model = YOLO('./detect/train17/weights/best.pt')

project = ls.get_project(25)
task_ids = project.get_unlabeled_tasks_ids() # Does it include skipped ones -> yes unfortunately

"""
results = model.predict("test_dataset/images/train/0007044.png", conf=0.8)
for result in results:
    # Detection
    result.boxes.xywh   # box with xywh format, (N, 4)
    result.boxes.cls
    result.boxes.conf
    print(result.names[result.boxes.cls.cpu().numpy()[0]])
    print(result.boxes.xywh.cpu().numpy())
    print(result.boxes.conf.cpu().numpy()[0])
    
quit()
"""
for task in task_ids:
    task_output = project.get_task(task)
    image_file_name = task_output["storage_filename"]
    download_from_minio(project, image_file_name, "/tmp/")
    image_path = Path("/tmp/") / image_file_name
    results = model.predict(image_path, conf=0.8)
    
    # TODO what about multiple boxes?
    for result in results:  
        if result.boxes.xywh.nelement() > 0:
            x = result.boxes.xywh.cpu().numpy()[0][0] / result.orig_img.shape[1] * 100
            y = result.boxes.xywh.cpu().numpy()[0][1] / result.orig_img.shape[0] * 100
            w = result.boxes.xywh.cpu().numpy()[0][2] / result.orig_img.shape[1] * 100
            h = result.boxes.xywh.cpu().numpy()[0][3] / result.orig_img.shape[0] * 100
            print()
            print(x ,y, w, h)
            ls_result = {"result": [{'type': 'rectanglelabels', 'value': {'x': x, 'y': y, 'width': w, 'height': h, 'rotation': 0, 'rectanglelabels': [result.names[result.boxes.cls.cpu().numpy()[0]]]}, 'to_name': 'image', 'from_name': 'label'}]}
            a = project.create_annotation(task, **ls_result)

