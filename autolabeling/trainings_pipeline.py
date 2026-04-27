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
import yaml

def log_custom_data(trainer, filename):
    mlflow.log_artifact(filename, artifact_path="dataset")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Pfad zur config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    target_class = config.get("target_class", "Ball")
    modelsize = config["modelsize"]
    camera = str(config["camera"]).upper()
    log_ids = config["log_ids"]
    epochs = config["epochs"]
    split_ratio = config.get("split_ratio", 0.8)
    seed = config.get("seed", random.randint(1, 1000000))

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
    run_path = f"runs/{target_class}/{camera}/{run_timestamp}_{modelsize}"

    os.makedirs(run_path, exist_ok=True)

    """
    Create Dataset File
    """
    dataset_file_name = f"{run_path}/ds_{camera}_{'-'.join(map(str, log_ids))}_{run_timestamp}_{modelsize}.json"
    print(f"The seed for this run: {seed}")
    create_dataset_json(log_ids, camera, target_class, v_client,l_client, dataset_file_name, split_ratio, seed)
    create_local_yolo_ds(dataset_file_name, run_path, target_class)

    """
    Train Yolo Model
    """
    mlflow.set_tracking_uri("https://mlflow.berlin-united.com/")
    os.environ["MLFLOW_EXPERIMENT_NAME"] = f"{target_class}-{camera}-classifier"
    
    # Load the YOLO26 model (n=nano, s=small, m=medium, l=large, x=extra-large)
    model = YOLO(f"yolo26{modelsize}.pt")
    log_callback = partial(log_custom_data, filename=dataset_file_name)
    model.add_callback("on_train_end", log_callback)
    
    mlflow.set_experiment(f"{target_class}-{camera}-classifier")

    mlflow.log_param("target_class", target_class)
    mlflow.log_param("user", os.environ.get("MLFLOW_USER"))
    mlflow.log_param("split_seed", seed)
    mlflow.log_param("split_ratio", split_ratio)
    target_project = os.path.abspath(f"{run_path}")
    target_name=f"autolabel_model"
    
    results = model.train(
        data=f"{run_path}/dataset.yaml", 
        epochs=epochs, 
        imgsz=640, 
        optimizer="MuSGD",
        batch=16, # Auto-determines best batch size for your GPU
        project=target_project,
        name=target_name, 
        exist_ok=True         
    )

    project_id = get_project_id(v_client, log_ids[0], camera)
    print(f"\nTraining complete. To autolabel new images, run:")
    print(f"Trage den Modell-Pfad '{target_project}/{target_name}/weights/best.pt' in deine config.yaml ein.")
    print(f"Starte danach: uv run run_model_yolo.py -c config.yaml")