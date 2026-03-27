from vaapi.client import Vaapi
from functools import partial
from label_studio_sdk import LabelStudio
from tools import create_dataset_json, create_local_yolo_ds
from ultralytics import YOLO
import mlflow
import datetime
import os

def log_custom_data(trainer, filename):
    mlflow.log_artifact(filename, artifact_path="dataset")

if __name__ == "__main__":
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
    """
    Create Dataset File
    """
    log_ids = [679]
    camera="BOTTOM"
    dataset_file_name = f"ball_detection_ds_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.json"
    create_dataset_json(log_ids, camera, v_client,l_client, dataset_file_name)
    create_local_yolo_ds(dataset_file_name, output_path=f"datasets/ball_detection_ds_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}", split_ratio=0.8)

    """
    Train Yolo Model
    """
    mlflow.set_tracking_uri("https://mlflow.berlin-united.com/")
    os.environ["MLFLOW_EXPERIMENT_NAME"] = f"GO26-Autolabeling Model-{camera}"
    
    # Load the YOLO26 model (n=nano, s=small, m=medium, l=large, x=extra-large)
    model = YOLO("yolo26n.pt")
    log_callback = partial(log_custom_data, filename=dataset_file_name)
    model.add_callback("on_train_end", log_callback)
    
    mlflow.set_experiment(f"GO26-Autolabeling Model-{camera}")

    mlflow.log_param("user", os.environ.get("MLFLOW_USER"))
    ziel_projekt = os.path.abspath(f"data/{camera}/autolabel_model")
    ziel_name=f"yolo_{camera}_run_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    
    results = model.train(
        data=f"dataset.yaml", 
        epochs=500, 
        imgsz=640, 
        optimizer="MuSGD",
        batch=-1, # Auto-determines best batch size for your GPU
        project=ziel_projekt,
        name=ziel_name, 
        exist_ok=True         
    )
