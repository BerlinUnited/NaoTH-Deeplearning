# YOLO based pre annotations for Nao Images

## Download data and create a dataset for training
```
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

## Run a docker container with access to the GPUs
```
docker run -it --privileged -u $(id -u):$(id -g) --cpuset-cpus="4-16" -v ${PWD}:/usr/src/datasets -v ${PWD}:/usr/src/ultralytics/runs/ --gpus all --ipc host ultralytics/ultralytics:8.1.42-python /bin/bash
docker run -it --privileged -u $(id -u):$(id -g) -v ${PWD}:/usr/src/datasets -v ${PWD}:/usr/src/ultralytics/runs/ --gpus all --ipc host ultralytics/ultralytics:8.1.42-python /bin/bash
docker run -it --privileged --cpuset-cpus="4-16" -v ${PWD}:/usr/src/datasets -v ${PWD}:/usr/src/ultralytics/runs/ --gpus all --ipc host ultralytics/ultralytics:8.1.42 /bin/bash
docker run -it --privileged -v ${PWD}:/usr/src/datasets -v ${PWD}:/usr/src/ultralytics/runs/ --gpus all --ipc host ultralytics/ultralytics:latest /bin/bash
```

cd /usr/src/datasets
python -m pip install -r requirements.txt (for labelstudio sdk which is used later for annotating live)
yolo train data=test_dataset.yaml model=yolov8n.pt epochs=1200 lr0=0.01

yolo train data=test_dataset.yaml model=yolov8s.pt epochs=1200 lr0=0.01

python train.py -m 2024-04-01-yolov8s-best.pt -ds yolo-full-size-detection_dataset_2024-04-04.yaml

tensorboard --logdir /usr/src/ultralytics/runs/ --bind-all &

## Further ideas
It is technically possible to trigger the training whenever annotations change -> it comes out of the box for enterprise but I could also reimplement this
configure trainings params
maybe use larger yolo base model
test that segmentations would not destroy stuff