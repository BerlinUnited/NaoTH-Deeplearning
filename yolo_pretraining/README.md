# YOLO based pre annotations for Nao Images
We provide a docker image and scripts for training a ultralytics YOLOv8 model on our full images. The images are are downloaded from our minio server and the annotation are downloaded from our Labelstudio instance. You need to configure access for it.

## Set up your environment
You can either create a python virtual env or use the provided docker image. See below for instructions for both. In either case you need to configure some environment variables. The values are in our [internal wiki](https://scm.cms.hu-berlin.de/berlinunited/orga/-/wikis/team/Accounts). More information about the environment variables can be found in our [k8s repo](https://scm.cms.hu-berlin.de/berlinunited/projects/k8s-cluster).

You can run each line or put them in the `.bashrc` file
```
export DB_PASS=
export MINIO_PASS=
export LS_URL=
export LS_KEY=
export REPL_ROOT=
```

You need to mount the folder `/vol/repl261-vol4/naoth` from `gruenau10.informatik.hu-berlin.de` via sshfs. Set the REPL_ROOT variable to the mount point. Scripts here will access the subfolders models and datasets for uploads. The folders are also publicly available on the web as [models.naoth.de](models.naoth.de) and [datasets.naoth.de](datasets.naoth.de) in read only mode. A tutorial for setting up sshfs mounts can be found in our internal wiki.

### Setup python environment
```
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

### Setup docker environment
If you run on your own system you can run it like this:
```bash
docker run -it --privileged -v ${PWD}:/usr/src/datasets -v ${PWD}:/usr/src/ultralytics/runs/ --gpus all --ipc host scm.cms.hu-berlin.de:4567/berlinunited/tools/naoth-deeplearning/yolo_image:latest /bin/bash
```
The `--privileged`, `--gpus all` and `--ipc host` flags together make sure that the docker container can access the GPU and that the use of the CPU is not limited by the OS.

The `-v ${PWD}:/usr/src/datasets -v ${PWD}:/usr/src/ultralytics/runs/` flags mount the current path in which the docker command was run is mounted under `/usr/src/datasets` and `/usr/src/ultralytics/runs/`. The makes sure that the output of the training is saved in the current working directory on the host and that the current working directory on the host is the same as in the docker container. The ultralytics image expects training to be started in `/usr/src/datasets` and that the datasets are in that directory.

!!! If you are using this on servers owned by the team please be conscious of others using the servers too. For this use the additional flags `-u $(id -u):$(id -g) --cpuset-cpus="4-14"` so the total command is 
```bash
docker run -it --privileged -u $(id -u):$(id -g) --cpuset-cpus="4-14" -v ${PWD}:/usr/src/datasets -v ${PWD}:/usr/src/ultralytics/runs/ --gpus all --ipc host scm.cms.hu-berlin.de:4567/berlinunited/tools/naoth-deeplearning/yolo_image:latest /bin/bash
```
`-u $(id -u):$(id -g)` make sure commands inside the docker container have the same user and group id as you have on the server. This makes it possible to know who started a training with `htop`. `--cpuset-cpus="4-14"` limits the cpu cores you can use. Please adjust this accordingly. As a rule please leave half of the cores for others. Also coordinate with other people training models.

## Run inference
```
python run_model_in_ls.py -m <model name> -p <project id> <project id> <project id>
```
Have a look at [https://models.naoth.de/](https://models.naoth.de/) for a list of available models. For example you can choose `2024-04-06-yolov8s-best-top.pt`
If the model file is not present in the current working dir it will be downloaded.

## Run training
```
python train.py -ds <dataset name> -m <basemodel> -c Top -u Stella Alice
```
# python train_top.py -ds yolo-full-size-detection_dataset_top_2024-04-08.yaml -m yolov8n -c Top -u "Stella Alice"
A dataset argument needs to be the path to the yaml file. The model argument should either be yolov8n or yolov8s


## Create new datasets
If you want to create datasets and upload them then you need to make sure you have mounted the correct repl folder with sshfs.
TODO: add more information here.

## Build Custom YOLO Image
The pipeline already builds the image. You can pull that with
```bash
docker pull scm.cms.hu-berlin.de:4567/berlinunited/tools/naoth-deeplearning/yolo_image:latest
```
or you can built it locally with
```bash
docker build -t yolo_image:latest .
```

## Run a docker container with access to the GPUs


cd /usr/src/datasets
python -m pip install -r requirements.txt (for labelstudio sdk which is used later for annotating live)
yolo train data=test_dataset.yaml model=yolov8n.pt epochs=1200 lr0=0.01

yolo train data=test_dataset.yaml model=yolov8s.pt epochs=1200 lr0=0.01

python train.py -m 2024-04-01-yolov8s-best.pt -ds yolo-full-size-detection_dataset_2024-04-04.yaml

tensorboard --logdir /usr/src/ultralytics/runs/ --bind-all &

