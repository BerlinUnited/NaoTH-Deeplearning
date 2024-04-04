from ultralytics import YOLO
from ultralytics import settings
import mlflow
from mflow_callbacks import on_pretrain_routine_end, on_train_epoch_end, on_fit_epoch_end, on_train_end

# disable the mlfow integration from ultralytics because you can't override the callbacks and in the end it tries to upload the model with a single post requests without splitting (very weird)
# this leads to an request entity to large error from the loadbalancer -> should be investigated at some point
settings.update({'mlflow': False})
#os.environ['MLFLOW_TRACKING_URI'] = "https://mlflow.berlinunited-cloud.de/"
#os.environ['MLFLOW_EXPERIMENT_NAME'] = "YOLOv8 Full Size"
mlflow.set_tracking_uri("https://mlflow.berlinunited-cloud.de/")
mlflow.set_experiment("YOLOv8 Full Size")

with mlflow.start_run() as run:
    # Load a model
    #model = YOLO('detect/train4/weights/best.pt')  # load a pretrained model (recommended for training)
    model = YOLO('yolov8n.pt')
    # Train the model
    # have to reimplement all the mlflow callbacks myself so that I can change on of them
    model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)


    results = model.train(data="test_dataset.yaml", epochs=1, patience=50)

    # Log parameters, metrics, and model
    #mlflow.log_params({"epochs": 1200, "model": "yolov8s.pt", "data": "test_dataset.yaml"})
    #1mlflow.log_metrics({"mAP_0.5": results.map50, "mAP_0.5:0.95": results.map})
    #mlflow.pytorch.log_model(model, "model") does not work yet: have a look at https://github.com/mlflow/mlflow/issues/7820

    # End the run
    #mlflow.end_run()

# TODO upload the model (maybe only if its better?)
# TODO we need to name the model for sure here