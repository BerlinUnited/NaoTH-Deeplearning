"""
Checks if models are available in mlflow for the given project and if not falls back on a default model
"""
from ultralytics import YOLO
from pathlib import Path
import os
import sys

helper_path = os.path.join(os.path.dirname(__file__), "../../tools")
sys.path.append(helper_path)

from helper import get_file_from_server


def get_best_yolo_model_mlflow():
    """
    Not implemented yet
    """
    pass


def get_yolo_model(camera: str, model:str):
    # TODO get the best performing model from mlflow and only if there is none use those old models
    # TODO use pathlib here
    if model:
        model_name = model
        class_mapping = None
    else:
        # Otherwise, use the camera-specific default
        model_name = {
            "BOTTOM": "2024-04-16-yolov8s-bottom-resilient-eel-775.pt",
            "TOP": "2024-04-27-yolov8s-top-indecisive-snake-51.pt"
        }.get(camera)

        # HACK we just added it in manually - will no longer be needed if we train on the db data
        
        class_mapping = {
            # yolo - db
            0: 2,
            1: 1,
            2: 3,
        }
        
        if model_name is None:
            raise ValueError(f"Unknown camera: {camera}")

    model_path = f"./models/{model_name}"
    
    if not os.path.isfile(model_path):
        get_file_from_server(f"https://models.naoth.de/{model_name}", model_path)



    return YOLO(model_path), class_mapping
    