import mlflow
from pathlib import Path
import os
import sys
import requests
from requests.auth import HTTPBasicAuth


def on_pretrain_routine_end(trainer):
    from ultralytics.utils import LOGGER, colorstr

    PREFIX = colorstr("MLflow: ")
    SANITIZE = lambda x: {k.replace("(", "").replace(")", ""): float(v) for k, v in x.items()}
    uri = mlflow.get_tracking_uri()
    LOGGER.debug(f"{PREFIX} tracking uri: {uri}")
    mlflow.set_tracking_uri(uri)

    # Set experiment and run names
    # experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME") or trainer.args.project or "/Shared/YOLOv8"
    run_name = os.environ.get("MLFLOW_RUN") or trainer.args.name
    # mlflow.set_experiment(experiment_name)

    mlflow.autolog()
    try:
        active_run = mlflow.active_run() or mlflow.start_run(run_name=run_name)
        LOGGER.info(f"{PREFIX}logging run_id({active_run.info.run_id}) to {uri}")
        if Path(uri).is_dir():
            LOGGER.info(f"{PREFIX}view at http://127.0.0.1:5000 with 'mlflow server --backend-store-uri {uri}'")
        LOGGER.info(f"{PREFIX}disable with 'yolo settings mlflow=False'")
        mlflow.log_params(dict(trainer.args))
    except Exception as e:
        LOGGER.warning(f"{PREFIX}WARNING ⚠️ Failed to initialize: {e}\n" f"{PREFIX}WARNING ⚠️ Not tracking this run")


def on_train_epoch_end(trainer):
    from ultralytics.utils import colorstr

    SANITIZE = lambda x: {k.replace("(", "").replace(")", ""): float(v) for k, v in x.items()}
    """Log training metrics at the end of each train epoch to MLflow."""
    if mlflow:
        mlflow.log_metrics(
            metrics={
                **SANITIZE(trainer.lr),
                **SANITIZE(trainer.label_loss_items(trainer.tloss, prefix="train")),
            },
            step=trainer.epoch,
        )


def on_fit_epoch_end(trainer):
    from ultralytics.utils import colorstr

    SANITIZE = lambda x: {k.replace("(", "").replace(")", ""): float(v) for k, v in x.items()}
    """Log training metrics at the end of each fit epoch to MLflow."""
    if mlflow:
        mlflow.log_metrics(metrics=SANITIZE(trainer.metrics), step=trainer.epoch)


def on_train_end(trainer):
    """Log model artifacts at the end of the training."""
    pass
    # if mlflow:
    # mlflow.log_artifact(str(trainer.best.parent))  # log save_dir/weights directory with best.pt and last.pt
    # for f in trainer.save_dir.glob("*"):  # log all other files in save_dir
    #    if f.suffix in {".png", ".jpg", ".csv", ".pt", ".yaml"}:
    #        mlflow.log_artifact(str(f))
    # keep_run_active = os.environ.get("MLFLOW_KEEP_RUN_ACTIVE", "False").lower() == "true"
    # if keep_run_active:
    #    LOGGER.info(f"{PREFIX}mlflow run still alive, remember to close it using mlflow.end_run()")
    # else:
    #    mlflow.end_run()
    #    LOGGER.debug(f"{PREFIX}mlflow run ended")


#       LOGGER.info(
#          f"{PREFIX}results logged to {mlflow.get_tracking_uri()}\n"
#         f"{PREFIX}disable with 'yolo settings mlflow=False'"
#    )


def set_tracking_url(url="https://mlflow.berlin-united.com/", fail_on_timeout=False):
    try:
        # we can either get an error or an undesireable status code. Check for both
        if os.environ.get('MLFLOW_TRACKING_USERNAME') is not None:
            page = requests.get(url, 
                            auth=HTTPBasicAuth(os.environ.get("MLFLOW_TRACKING_USERNAME"), 
                                                os.environ.get("MLFLOW_TRACKING_PASSWORD")
                                                ),
                            timeout=10)        
        else:
            page = requests.get(url, timeout=10)
        if page.status_code == 200:
            mlflow.set_tracking_uri(url)
        else:
            print(f"Error connecting to mlflow. Can't upload trainings progress to {url}")
            if fail_on_timeout:
                sys.exit(1)
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError):
        print(f"Error connecting to mlflow. Can't upload trainings progress to {url}")
        if fail_on_timeout:
            sys.exit(1)
