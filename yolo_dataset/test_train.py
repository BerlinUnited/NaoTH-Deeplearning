import sys
import os

helper_path = os.path.join(os.path.dirname(__file__), "../")
sys.path.append(helper_path)

from ultralytics import YOLO, settings

import argparse
import numpy as np
import torch.nn as nn
from os import environ
import shutil
from zipfile import ZipFile
from pathlib import Path



class Dummymodel(nn.Module):
    def __init__(self):
        super().__init__()



def start_train():
    model = YOLO("2024-04-14-yolov8n-best-top-delicate-crow-184.pt")

    # Train the model
    # pytorch warning says it will create 20 workers for val this is because the val workers are always twice of the worker argument or if none given twice of what it calculated is the max
    results = model.train(
        data="test_ball_detection50.yaml",
        epochs=20,
        batch=4,
        patience=100,
        workers=10,
        name="test_run_em",
        verbose=True,
    )

    


if __name__ == "__main__":
    
    start_train()

    # TODO I could use the run name to name the model and upload it
