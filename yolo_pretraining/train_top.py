import sys
import os

helper_path = os.path.join(os.path.dirname(__file__), '../tools')
sys.path.append(helper_path)

from ultralytics import YOLO, settings
import mlflow
import argparse
import numpy as np

from mflow_callbacks import on_pretrain_routine_end, on_train_epoch_end, on_fit_epoch_end, on_train_end
from helper import get_file_from_server

# disable the mlfow integration from ultralytics because you can't override the callbacks and in the end it tries to upload the model with a single post requests without splitting (very weird)
# this leads to an request entity to large error from the loadbalancer -> should be investigated at some point
settings.update({'mlflow': False})
mlflow.set_tracking_uri("https://mlflow.berlinunited-cloud.de/")
mlflow.set_experiment("YOLOv8 Full Size - Top")

with mlflow.start_run() as run:
    # Load a model
    #model = YOLO('detect/train4/weights/best.pt')  # load a pretrained model (recommended for training)
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model")
    parser.add_argument("-ds", "--dataset")
    args = parser.parse_args()
    if args.model:
        print(f"will use specified model for further training: https://models.naoth.de/{args.model}")
        get_file_from_server(f"https://models.naoth.de/{args.model}", args.model)
        model = YOLO(args.model)
    else:
        print("will use base model from ultralytics for training")
        model = YOLO('yolov8n.pt')

    dummy_dataset = mlflow.data.from_numpy(np.array([]), targets=np.array([]), source=f"https://datasets.naoth.de/{args.dataset}", name=args.dataset)
    mlflow.log_input(dummy_dataset, context="training", tags={"name": args.dataset})

    # Train the model
    # have to reimplement all the mlflow callbacks myself so that I can change on of them
    model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)

    results = model.train(data=args.dataset, epochs=1200, patience=50, workers=10)

    # Log parameters, metrics, and model
    #mlflow.log_params({"epochs": 1200, "model": "yolov8s.pt", "data": "test_dataset.yaml"})
    #1mlflow.log_metrics({"mAP_0.5": results.map50, "mAP_0.5:0.95": results.map})
    #mlflow.pytorch.log_model(model, "model") does not work yet: have a look at https://github.com/mlflow/mlflow/issues/7820
    # TODO maybe we can make a hack here and create a dummy model with the correct name and metadata pointing to the correct model

    # End the run
    #mlflow.end_run()

# TODO upload the model (maybe only if its better?)
# TODO we need to name the model for sure here