import sys
import os

helper_path = os.path.join(os.path.dirname(__file__), '../tools')
sys.path.append(helper_path)

from ultralytics import YOLO, settings

import mlflow
import argparse
import numpy as np
import torch.nn as nn
from os import environ
import shutil
from zipfile import ZipFile
from pathlib import Path
from mflow_helper import on_pretrain_routine_end, on_train_epoch_end, on_fit_epoch_end, on_train_end, set_tracking_url

class Dummymodel(nn.Module):
    def __init__(self):
        super().__init__()


def check_repl_access():
    repl_root = environ.get("REPL_ROOT")
    if repl_root:
        repl_root = Path(repl_root)
        if not repl_root.exists():
            print("ERROR REPL_ROOT does not point to a folder that exists")
            quit()
    else:
        print("ERROR REPL_ROOT is not defined")
        quit()


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

        run = mlflow.active_run()

        # Train the model
        # pytorch warning says it will create 20 workers for val this is because the val workers are always twice of the worker argument or if none given twice of what it calculated is the max
        results = model.train(data=Path("datasets") / args.dataset, epochs=500, batch=32, patience=100, workers=10, name=run.info.run_name, verbose=True, device=0)
        
        # upload the model here
        # FIXME this prevents using this on deepl hardware
        model_name = f"{args.model}-{args.camera}-{run.info.run_name}.pt"
        local_model_path = Path("detect") / Path(run.info.run_name) / "weights" / "best.pt"
        remote_model_path = Path(environ.get("REPL_ROOT")) / "models" / model_name
        
        shutil.copyfile(local_model_path, remote_model_path)

        # we make a hack here and create a dummy model with the correct name and metadata pointing to the correct model (TODO: document how to find the correct model)
        dummy_model = Dummymodel()
        mlflow.pytorch.log_model(dummy_model, "yolov8n.pt", registered_model_name=f"{args.model}-{args.camera}")
        

        # End the run
        mlflow.end_run()

    # TODO upload the model (maybe only if its better?)
    # TODO we need to name the model for sure here

if __name__ == "__main__":
    check_repl_access()
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
    settings.update({"runs_dir": Path("./mlruns").resolve()})
    settings.update({"datasets_dir": Path("./").resolve()})
    
    # set up remote tracking if the mlflow tracking server is available
    set_tracking_url()

    mlflow.set_experiment(f"YOLOv8 Full Size - {args.camera.capitalize()}")
    mlflow.enable_system_metrics_logging()

    start_train(args)

    # TODO I could use the run name to name the model and upload it