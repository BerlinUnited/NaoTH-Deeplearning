# Train Yolo models for use inside CVAT
This folder contains code for training on fullsize images from our CVAT server (https://ball.informatik.hu-berlin.de/).


## Windows Setup
You need to install WSL2 and docker first and enable the WSL2 backend in the docker settings. Install a current Cuda
driver from https://developer.nvidia.com/cuda/wsl

To test if Cuda works in WSL you can run:
```bash
docker run --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```

## Setup in NaoTH Lab
Lab PC has RTX A4000
installed the nvidia graphics driver from the graphical software updater that comes with ubuntu

followed the installation instructions from
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/

container toolkit installation
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

# Get Datasets from CVAT
TODO: download it
TODO convert it to ultralytics format

# Train in docker container 
docker run -it --privileged -v ${PWD}:/work --gpus all --ipc host ultralytics/ultralytics:latest /bin/bash
yolo detect train model=yolov8n.pt data=coco128.yaml device=cpu



