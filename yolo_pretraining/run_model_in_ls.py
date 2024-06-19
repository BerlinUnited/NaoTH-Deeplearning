"""
    run the best model on new data in labelstudio
"""

import sys
import os

helper_path = os.path.join(os.path.dirname(__file__), "../tools")
sys.path.append(helper_path)
import uuid
from pathlib import Path
from label_studio_sdk import Client
from minio import Minio
from ultralytics import YOLO
import argparse

from helper import get_file_from_server

LABEL_STUDIO_URL = "https://ls.berlin-united.com/"
API_KEY = "6cb437fb6daf7deb1694670a6f00120112535687"
ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
ls.check_connection()

mclient = Minio(
    "minio.berlin-united.com",
    access_key="naoth",
    secret_key="HAkPYLnAvydQA",
)


def download_from_minio(project, filename, output_folder):
    bucket_name = project.title
    output = Path(output_folder) / filename
    mclient.fget_object(bucket_name, filename, output)


if __name__ == "__main__":
    # TODO use argparse for setting the model for now, later maybe we can utilize mlflow to automatically select the best model and download it?
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model")
    parser.add_argument(
        "-p", "--project", nargs="+", help="Labelstudio project ids separated by a space", required=True
    )
    args = parser.parse_args()

    print(f"will download https://models.naoth.de/{args.model}")
    get_file_from_server(f"https://models.naoth.de/{args.model}", args.model)

    # load the current best model
    model = YOLO(args.model)

    for id in args.project:
        project = ls.get_project(int(id))
        task_ids = project.get_unlabeled_tasks_ids()
        print(f"Annotating project {id}")
        for task in task_ids:
            task_output = project.get_task(task)
            # FIXME: this only works with minio as backend for now
            image_file_name = task_output["storage_filename"]
            download_from_minio(project, image_file_name, "/tmp/")
            image_path = Path("/tmp/") / image_file_name
            results = model.predict(image_path, conf=0.8, verbose=False)

            for result in results:
                if result.boxes.xywh.nelement() > 0:
                    label_studio_result_list = list()
                    num_bbox = len(result.boxes.xywh.cpu().numpy())
                    for idx, box in enumerate(result.boxes.xywh.cpu().numpy()):
                        x = box[0] - box[2] / 2  # x and y of the output are the center coordinates
                        y = box[1] - box[3] / 2
                        w = box[2] / result.orig_img.shape[1] * 100
                        h = box[3] / result.orig_img.shape[0] * 100
                        x = x / result.orig_img.shape[1] * 100
                        y = y / result.orig_img.shape[0] * 100

                        # create one annotation with multiple results
                        single_ls_result = {
                            "id": uuid.uuid4().hex[:9].upper(),
                            "type": "rectanglelabels",
                            "value": {
                                "x": x,
                                "y": y,
                                "width": w,
                                "height": h,
                                "rotation": 0,
                                "rectanglelabels": [result.names[result.boxes.cls.cpu().numpy()[idx]]],
                            },
                            "to_name": "image",
                            "from_name": "label",
                            "origin": "manual",
                            "original_width": result.orig_img.shape[1],
                            "original_height": result.orig_img.shape[0],
                        }
                        label_studio_result_list.append(single_ls_result)

                    print(f"\tadd {num_bbox} bounding boxes")
                    ls_result = {"result": label_studio_result_list, "completed_by": 2}
                    a = project.create_annotation(task, **ls_result)
                else:
                    print(f"\tadd empty annotation here")
                    ls_result = {"result": [], "completed_by": 2}
                    a = project.create_annotation(task, **ls_result)

# 639
