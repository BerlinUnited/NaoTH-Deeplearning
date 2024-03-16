# YOLO based pre annotations for Nao Images

## Download data and create a dataset for training
```
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

## Run a docker container with access to the GPUs
```
docker run -it --privileged -v ${PWD}:/work -v ${PWD}:/usr/src/ultralytics/runs/ --gpus all --ipc host ultralytics/ultralytics:latest /bin/bash
```

cd /usr/src/datasets
yolo train data=test_dataset.yaml model=yolov8n.pt epochs=100 lr0=0.01