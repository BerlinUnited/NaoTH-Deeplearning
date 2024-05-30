# NaoTH-DeepLearning
This repo contains tools for datasets, learning, compiling and evaluating neural networks. This is for all deep learning done as part of the RoboCup SPL efforts of the Berlin United Team.

Each folder corresponds to one project. A project might be whistle detection, ball detection on patches, yolo ball detection, object detection on full images, etc. Commonly used functions should go into the tools folder.

## Installation
We prepared different requirement files if you want to work with the tensorflow projects or the pytorch projects. Choose the corresponding requirements file during installation.

```bash
python3 -m venv venv_tf
source venv_tf/bin/activate
python -m pip install -r requirements-tensorflow.txt
```
or 
```bash
python3 -m venv venv_pytorch
source venv_pytorch/bin/activate
python -m pip install -r requirements-tensorflow.txt
```