# Referee Detection
According to the current rules of RoboCup SPL in 2025 we need to detect the referee and its pose.
> https://spl.robocup.org/wp-content/uploads/SPL-Rules-master.pdf

This project is only concerned with the pose detection of the referee. We expect an image in which the referee is already isolated. For that we use the referee detector model.

## Setup
TODO: ssh setup and access to our universities server
TODO: dvc setup
TODO: check out https://marketplace.visualstudio.com/items?itemName=Iterative.dvc
TODO: describe and test how to switch between different versions of the datasets: https://dvc.org/doc/start 
TODO: test the transformer models from huggingface

## Run Inference


We use different models for detecting poses. Large models should be used for pre labeling and run on full images and a smaller model should run eventually on the Nao robot.

TODO: check that each model outputs the same data  
TODO: we are planning on using movenet for Nao inference: https://www.tensorflow.org/hub/tutorials/movenet


## Get or create dataset
We use DVC for versioning our dataset. If a dataset is already created you can just run
```
dvc pull
```

## Train new referee detection model