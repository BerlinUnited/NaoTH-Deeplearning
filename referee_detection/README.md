# Referee Detection
According to the current rules of RoboCup SPL in 2025 we need to detect the referee and its pose.
> https://spl.robocup.org/wp-content/uploads/SPL-Rules-master.pdf

This project is only concerned with detection the referee in full size images so we can later use the bounding box information to improve the accuracy of any pose detection mode.

## Setup
TODO: ssh setup and access to our universities server
TODO: dvc setup
TODO: check out https://marketplace.visualstudio.com/items?itemName=Iterative.dvc
TODO: describe and test how to switch between different versions of the datasets: https://dvc.org/doc/start 
TODO ultralytics setup

## Run Inference
We use Yolo11 from Ultralytics for detecting the referee
We use DVC for versioning our dataset. If a dataset is already created you can just run
```
dvc pull
```
other have a look at the next section.

## Create dataset
TODO create a dataset from all frames in the gamecontroller state where they should see the referee

## Train new referee detection model
TODO