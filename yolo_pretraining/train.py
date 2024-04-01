from ultralytics import YOLO
from ultralytics import settings
import mlflow
import os
import re

settings.update({'mlflow': True})
os.environ['MLFLOW_TRACKING_URI'] = "https://mlflow.berlinunited-cloud.de/"
os.environ['MLFLOW_EXPERIMENT_NAME'] = "YOLOv8 Full Size"
#mlflow.set_tracking_uri("https://mlflow.berlinunited-cloud.de/")
#mlflow.set_experiment("YOLOv8 Full Size")

def on_fit_epoch_end(trainer):
    print('in the on_fit_epoch_end')
    metrics_dict = {f"{re.sub('[()]', '', k)}": float(v) for k, v in trainer.metrics.items()}
    mlflow.log_metrics(metrics=metrics_dict, step=trainer.epoch)

with mlflow.start_run() as run:
    # Load a model
    #model = YOLO('detect/train4/weights/best.pt')  # load a pretrained model (recommended for training)
    model = YOLO('yolov8n.pt')
    # Train the model
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    results = model.train(data="test_dataset.yaml", epochs=2, patience=50)

    # Log parameters, metrics, and model
    #mlflow.log_params({"epochs": 1200, "model": "yolov8s.pt", "data": "test_dataset.yaml"})
    #mlflow.log_metrics({"mAP_0.5": results.map50, "mAP_0.5:0.95": results.map})
    #mlflow.pytorch.log_model(model, "model")

    # End the run
    mlflow.end_run()