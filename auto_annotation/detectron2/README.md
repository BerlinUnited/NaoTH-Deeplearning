# Auto Annotation for CVAT Data
For this project we use the detectron2 framework.

## Linux Setup
- TODO write setup for modern linux

## Windows Setup
The windows setup was tested on Windows 10 Pro 21H1 (OS build 19043.1348) equipped with a Nvidia GeForce RTX 3060. The
CUDA driver in version 11.2.2 () was installed from https://developer.nvidia.com/cuda-toolkit-archive because it's the 
tested CUDA version for tensorflow 2.7.0. The cuDNN version is 8.1, once again because it's recommended for tensorflow 2.7.0

Then the paths were setup according to the tensorflow docs https://www.tensorflow.org/install/gpu

It turns out that with the same driver setup pytorch 1.10.0 also works.

```bash
conda env create -f environment.yml
```

activate the created conda environment. You have to use the name specified in the environment.yml
```bash
conda activate test-detectron2
```
Install pytorch via pip because conda has problems installing the correct pytorch version because of channel order or something.
```bash
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## Train the network
- TODO: try new data in cvat format. Are there any changes that must be done manually?
- TODO setup training somewhere where its fast
- TODO: document the trainings code a bit


## Run inference locally
- TODO: write inference script for local execution
- TODO: move nuclio stuff to extra repo
- 

## Upload inferences to CVAT
- you can update annotations from jobs and tasks
  - can you get image data from a specific job?
  - test upload of annotations