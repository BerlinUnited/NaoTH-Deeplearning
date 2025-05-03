from ultralytics import YOLO, settings
from pathlib import Path

# TODO implement a model that only detects referees, later we will implement a model that can detect all bounding box classes

def start_train():

    # Train the model
    # pytorch warning says it will create 20 workers for val this is because the val workers are always twice of the worker argument or if none given twice of what it calculated is the max
    results = model.train(
        data=Path("datasets") / "referee_2025-02-02.yaml",
        epochs=10,
        batch=32,
        patience=100,
        workers=10,
        verbose=True,
        device=0,
    )



if __name__ == "__main__":
    # Load a model
    model = YOLO('yolo11n.pt')  # load a pretrained model (recommended for training)

    # disable the mlfow integration from ultralytics because you can't override the callbacks and in the end it tries to upload the model with a single post requests without splitting (very weird)
    # this leads to an request entity to large error from the loadbalancer -> should be investigated at some point


    # set up remote tracking if the mlflow tracking server is available
    settings.update({"runs_dir": str(Path("./mlruns").resolve())})
    settings.update({"datasets_dir": str(Path("./").resolve())})
    start_train()

    # TODO I could use the run name to name the model and upload it
