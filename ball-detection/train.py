"""Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
"""
import mlflow
import tensorflow as tf
import numpy as np
from models import fy_1500_new
import pickle
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError

mlflow.set_tracking_uri("https://mlflow.berlinunited-cloud.de/")
# TODO do something here once we have auth on the server


experiment_tags = {
    "user": "stella",
    "mlflow.note.content": "Tracking of ball detection progress",
}
my_experiment = mlflow.set_experiment("Ball-Detection2024")

def get_dataset_from_server(origin, target):
    # https://datasets.naoth.de/rc19_classification_16_bw_bottom.pkl
    def dl_progress(count, block_size, total_size):
        print('\r', 'Progress: {0:.2%}'.format(min((count * block_size) / total_size, 1.0)), sep='', end='', flush=True)

    if not Path(target).exists():
        target_folder = Path(target).parent
        target_folder.mkdir(parents=True, exist_ok=True)
    else:
        return

    error_msg = 'URL fetch failure on {} : {} -- {}'
    try:
        try:
            urlretrieve(origin, target, dl_progress)
            print('\nFinished')
        except HTTPError as e:
            raise Exception(error_msg.format(origin, e.code, e.reason))
        except URLError as e:
            raise Exception(error_msg.format(origin, e.errno, e.reason))
    except (Exception, KeyboardInterrupt):
        if Path(target).exists():
            Path(target).unlink()
        raise

model = fy_1500_new()
get_dataset_from_server("https://datasets.naoth.de/rc19_classification_16_bw_bottom.pkl", "rc19_classification_16_bw_bottom.pkl")




with open("rc19_classification_16_bw_bottom.pkl", "rb") as f:
    pickle.load(f)  # skip mean
    x = pickle.load(f)  # x are all input images
    y = pickle.load(f)  # y are the trainings target: [r, x,y,1]

    dataset = mlflow.data.from_numpy(x, targets=y, source="https://datasets.naoth.de/rc19_classification_16_bw_bottom.pkl", name="rc19_classification_16_bw_bottom.pkl")


with mlflow.start_run() as run:
    mlflow.set_experiment_tags(experiment_tags)
    mlflow.log_input(dataset, context="training", tags={"name": "rc19_classification_16_bw_bottom"})
    model.fit(x, y, batch_size=256, epochs=1, verbose=1, validation_split=0.1, callbacks=[mlflow.keras.MLflowCallback()])
    mlflow.tensorflow.log_model(model, artifact_path="fy_1500_new", registered_model_name="fy_1500_new")