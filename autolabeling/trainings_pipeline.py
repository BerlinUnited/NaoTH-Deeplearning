from vaapi.client import Vaapi
from functools import partial
from label_studio_sdk import LabelStudio
from tools import create_dataset_json, create_local_yolo_ds, get_log_ids_per_game, get_log_ids_per_event
from ultralytics import YOLO
import mlflow
import datetime
import os
import argparse
import random
import yaml


def log_custom_data(trainer, filename):
    """
    Implemented as callback here so that the artifact is attached to the run started by ultralytics
    """
    mlflow.log_artifact(filename, artifact_path="dataset")

def validate_config(config):
    if not config["target_class"]:
        raise ValueError("You must provide a target class in your config.")
    if not config["modelsize"]:
        raise ValueError("You must provide a modelsize in your config.")
    if not str(config["camera"]).upper():
        raise ValueError("You must provide a camera in your config.")
    if not config["epochs"]:
        raise ValueError("You must provide the epochs amount in your config.")
    if not log_ids and not ls_project_ids and not event_ids and not game_ids:
        raise ValueError(
            "You must provide either 'log_ids', 'game_ids', 'event_ids' or 'ls_project_ids' in your config."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Pfad zur config.yaml"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    validate_config(config)

    target_class = config["target_class"]
    modelsize = config["modelsize"]
    camera = str(config["camera"]).upper()
    epochs = config["epochs"]
    event_ids = config.get("event_ids")
    game_ids = config.get("game_ids")
    log_ids = config.get("log_ids")
    ls_project_ids = config.get("ls_project_ids")
    split_ratio = config.get("split_ratio", 0.8)
    seed = config.get("seed", random.randint(1, 1000000))

    if log_ids and ls_project_ids:
        print(
            "Both 'log_ids' and 'ls_project_id' are present in the config. Defaulting to 'log_ids'."
        )
        ls_project_ids = None

    if game_ids:
        log_ids = []
        for game_id in game_ids:
            log_ids += get_log_ids_per_game(game_id) 

    if event_ids:
        log_ids = []
        for event_id in event_ids:
            log_ids += get_log_ids_per_event(event_id) 

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

    run_timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_path = f"runs/{target_class}/{camera}/{run_timestamp}_{modelsize}"

    os.makedirs(run_path, exist_ok=True)

    """
    Create Dataset File
    """
    identifier = (
        "-".join(map(str, log_ids))
        if log_ids
        else "proj_" + "-".join(map(str, ls_project_ids))
    )
    dataset_file_name = (
        f"{run_path}/ds_{camera}_{identifier}_{run_timestamp}_{modelsize}.json"
    )

    create_dataset_json(
        log_ids,
        ls_project_ids,
        camera,
        target_class,
        v_client,
        l_client,
        dataset_file_name,
        split_ratio,
        seed,
    )
    create_local_yolo_ds(dataset_file_name, run_path, target_class)

    """
    Train Yolo Model
    """
    mlflow.set_tracking_uri("https://mlflow.berlin-united.com/")
    os.environ["MLFLOW_EXPERIMENT_NAME"] = f"{target_class}-{camera}-classifier-model"

    # Load the YOLO26 model (n=nano, s=small, m=medium, l=large, x=extra-large)
    model = YOLO(f"yolo26{modelsize}.pt")
    log_callback = partial(log_custom_data, filename=dataset_file_name)
    model.add_callback("on_train_end", log_callback)

    mlflow.set_experiment(f"{target_class}-{camera}-classifier-model")

    mlflow.log_param("target_class", target_class)
    mlflow.log_param("user", os.environ.get("MLFLOW_USER"))
    mlflow.log_param("split_seed", seed)
    mlflow.log_param("split_ratio", split_ratio)
    target_project = os.path.abspath(f"{run_path}")
    target_name = "autolabel_model"

    results = model.train(
        data=f"{run_path}/dataset.yaml",
        epochs=epochs,
        imgsz=640,
        optimizer="MuSGD",
        batch=-1,  # Auto-determines best batch size for your GPU
        project=target_project,
        name=target_name,
        exist_ok=True,
    )

    print("\nTraining complete.")
