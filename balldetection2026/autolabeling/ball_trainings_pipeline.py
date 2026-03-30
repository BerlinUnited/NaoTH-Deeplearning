"""
ball_trainings_pipeline.py
"""
from vaapi.client import Vaapi
from functools import partial
from label_studio_sdk import LabelStudio
from tools import create_dataset_json, create_local_yolo_ds, get_project_id
from ultralytics import YOLO
import mlflow
import datetime
import os
import argparse
import random

def log_custom_data(trainer, filename):
    mlflow.log_artifact(filename, artifact_path="dataset")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--modelsize", type=str, required=True, help="Set the yolo modelsize: n, m, l, x")
    parser.add_argument("-c", "--camera", type=str, required=True, help="Set the camera: BOTTOM or TOP")
    parser.add_argument("-l", "--log_ids", type=lambda s: s.split(","), required=True, help="Set the log ids: e.g. -l 678,679,683")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="Set the epochs number")
    parser.add_argument("-r", "--split_ratio", type=float, default=0.8, help="Set the split ratio for train and val sets (Default: 0.8)")
    parser.add_argument("-s", "--seed", type=int, default=random.randint(1, 1000000), help="Set the seed (Default: random seed)")
    args = parser.parse_args()

    camera = str(args.camera).upper()

    """
    Init API's
    """
    v_client = Vaapi(
        base_url=os.environ.get("VAT_API_URL"),
        api_key=os.environ.get("VAT_API_TOKEN"),
    )

    l_client = LabelStudio(
        base_url="https://labelstudio-api.berlin-united.com",
        api_key=os.environ.get("LABELSTUDIO_API_KEY"),
    )

    run_timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    run_path = f"runs/{camera}/{run_timestamp}_{args.modelsize}"

    os.makedirs(run_path, exist_ok=True)

    """
    Create Dataset File
    """
    dataset_file_name = f"{run_path}/ds_{camera}_{'-'.join(map(str, args.log_ids))}_{run_timestamp}_{args.modelsize}.json"
    print(f"The seed for this run: {args.seed}")
    create_dataset_json(args.log_ids, camera, v_client,l_client, dataset_file_name, args.split_ratio, args.seed)
    create_local_yolo_ds(dataset_file_name, run_path)

    """
    Train Yolo Model
    """
    mlflow.set_tracking_uri("https://mlflow.berlin-united.com/")
    os.environ["MLFLOW_EXPERIMENT_NAME"] = f"GO26-Autolabeling Model-{camera}"
    
    # Load the YOLO26 model (n=nano, s=small, m=medium, l=large, x=extra-large)
    model = YOLO(f"yolo26{args.modelsize}.pt")
    log_callback = partial(log_custom_data, filename=dataset_file_name)
    model.add_callback("on_train_end", log_callback)
    
    mlflow.set_experiment(f"GO26-Autolabeling Model-{camera}")

    mlflow.log_param("user", os.environ.get("MLFLOW_USER"))
    mlflow.log_param("split_seed", args.seed)
    mlflow.log_param("split_ratio", args.split_ratio)
    ziel_projekt = os.path.abspath(f"{run_path}")
    ziel_name=f"autolabel_model"
    
    results = model.train(
        data=f"{run_path}/dataset.yaml", 
        epochs=args.epochs, 
        imgsz=640, 
        optimizer="MuSGD",
        batch=-1, # Auto-determines best batch size for your GPU
        project=ziel_projekt,
        name=ziel_name, 
        exist_ok=True         
    )

    project_id = get_project_id(v_client, args.log_ids[0], camera)
    print(f"\nTraining complete. To autolabel new images, run:")
    print(f"uv run run_model_yolo.py -c {camera} -m {ziel_projekt}/{ziel_name}/weights/best.pt -p {project_id}")