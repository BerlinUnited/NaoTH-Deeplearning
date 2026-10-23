# /lustre/informatik/schlotts/yolo_balls_dataset

from ultralytics import YOLO
import mlflow
import os
# Load a lightweight model like YOLOv8n
model = YOLO("yolov8n.pt")


mlflow.set_tracking_uri("https://mlflow.berlin-united.com/")
mlflow_experiment_name = "booster_ball_model"
os.environ["MLFLOW_EXPERIMENT_NAME"] = mlflow_experiment_name

#log_callback = partial(log_custom_data, filename=dataset_file_name)
#model.add_callback("on_train_end", log_callback)

mlflow.set_experiment(mlflow_experiment_name)

# Train on your new dataset
results = model.train(
    data="/lustre/informatik/schlotts/yolo_balls_dataset/dataset.yaml",
    epochs=1000,
    imgsz=[544, 448], # super important to get this right
    optimizer="MuSGD",
    batch=-1,  # Auto-determines best batch size for your GPU
    exist_ok=True,
    scale=0.2, degrees=0.0,
    box=10,
)