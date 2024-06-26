# Ball Detection / Classification Models

This module contains several projects related to ball classification and detection (radius, ball_center).  
Each module comes with a `Dockerfile` and a `train.py` for reproducible training.

## Datasets

Each project is responsible for downloading the required datasets from https://datasets.naoth.de  
The `classifier_cnn` and `detector_cnn_ball_radius_center` projects both use the patch datasets we create in
`dataset_tools/ball*\*\_datasets`, which should already be uploaded to the datasets archive. See `dataset_tools/Readme.md` for more information and instructions on how to (re-)create these datasets.

## MlFlow Tracking

We track all training progress via mlflow at these tracking servers:

- https://mlflow.berlin-united.com/ (Default tracking)
- https://mlflow2.berlin-united.com/ (Intended for tracking from within DeepL infrastructure)

If you create a new project, please make sure to include the required mlflow tracking code.  
For implementation details, you can refer to `classifier_cnn/train.py`. Some of the most important things to track include:

- model parameters
- training metrics such as loss and accuracy
- artifacts such as trained model files and evaluation images

## Docker

All projects should contain a valid `Dockerfile` that runs the model training.  
If necessary, parametrize your training scripts so training can be done on the naoth goal server
or DeepL infrastructure.

TODO: Include information on how to ensure Dockerfile works on DeepL

## Classifier CNN (Ball/NoBall)

This project is used to train vanilla CNN models for ball classification on image patches.  
There are two ways to train a model:

- `train.py` allows you to train a single model instance, you set the model params as arguments to the script
- `train_*.sh` scripts will train multiple instantiations of a model over a combination of valid parameters

The `train_*.sh` are intended for high performance hardware, since they will run multiple model trainings when invoked.
It is recommended to do this via the included `Dockerfile` and adjusting the `CMD` or `ENTRYPOINT`.

You can use any of the following datasets to train a CNN classifier model:

- https://datasets.naoth.de/classification/

## Detector CNN (ball_radius, ball_center_x, ball_center_y) Regression

This project is used to train vanilla CNN models for Regression of ball location targets on image patches.  
There are two ways to train a model:

- `train.py` allows you to train a single model instance, you set the model params as arguments to the script
- `train_*.sh` scripts will train multiple instantiations of a model over a combination of valid parameters

The `train_*.sh` are intended for high performance hardware, since they will run multiple model trainings when invoked.
It is recommended to do this via the included `Dockerfile` and adjusting the `CMD` or `ENTRYPOINT`.

You can use any of the following datasets to train a CNN classifier model:

- https://datasets.naoth.de/detection/
