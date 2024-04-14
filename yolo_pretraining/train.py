import sys
import os

helper_path = os.path.join(os.path.dirname(__file__), '../tools')
sys.path.append(helper_path)

from ultralytics import YOLO, settings
import mlflow
import argparse
import numpy as np
import torch.nn as nn
from zipfile import ZipFile
from pathlib import Path
from mflow_callbacks import on_pretrain_routine_end, on_train_epoch_end, on_fit_epoch_end, on_train_end
from helper import get_file_from_server

class Dummymodel(nn.Module):
    def __init__(self):
        super().__init__()

dummy_model = Dummymodel()


def start_train(args):
    with mlflow.start_run() as run:
        print(f"will use {args.model} as base model from ultralytics for training")
        model = YOLO(f'{args.model}.pt')

        dummy_dataset = mlflow.data.from_numpy(np.array([]), targets=np.array([]), source=f"https://datasets.naoth.de/{args.dataset}", name=args.dataset)
        mlflow.log_input(dummy_dataset, context="training", tags={"name": args.dataset})

        # have to reimplement all the mlflow callbacks myself so that I can change on of them
        model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        model.add_callback("on_train_end", on_train_end)

        # Train the model
        results = model.train(data=Path("datasets") / args.dataset, epochs=1000, batch=32, patience=100, workers=12, verbose=True, device=0)

        mlflow.pytorch.log_model(dummy_model, "yolov8n.pt", registered_model_name="test") #does not work yet: have a look at https://github.com/mlflow/mlflow/issues/7820
        # TODO maybe we can make a hack here and create a dummy model with the correct name and metadata pointing to the correct model

        # End the run
        #mlflow.end_run()

    # TODO upload the model (maybe only if its better?)
    # TODO we need to name the model for sure here

if __name__ == "__main__":
    # Load a model
    #model = YOLO('detect/train4/weights/best.pt')  # load a pretrained model (recommended for training)
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-ds", "--dataset", required=True)
    parser.add_argument("-c", "--camera", required=True, choices=['bottom', 'top'])
    parser.add_argument("-u", "--user", required=True)
    args = parser.parse_args()

    os.environ["LOGNAME"] = args.user # needed because for now the docker container runs as root user

    # disable the mlfow integration from ultralytics because you can't override the callbacks and in the end it tries to upload the model with a single post requests without splitting (very weird)
    # this leads to an request entity to large error from the loadbalancer -> should be investigated at some point
    settings.update({'mlflow': False})
    mlflow.set_tracking_uri("https://mlflow.berlinunited-cloud.de/")
    mlflow.set_experiment(f"YOLOv8 Full Size - {args.camera.capitalize()}")
    mlflow.enable_system_metrics_logging()

    start_train(args)