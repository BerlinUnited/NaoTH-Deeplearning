## Train the ball model

### Setup Conda + environment
Conda is used since its easiest to set up with the dependencies. pytorch + pytorch vision + detectron2 needs to be compiled on the ball server to use cuda for training.

create folder for all the downloads:
`mkdir pytorch_source && cd pytorch_source`

Install anaconda
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
chmod +x Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh

# agree to everything. After logout and login the base env is activated 
```
-----------------------
Deactivate an environment: `conda deactivate`

Create an environment: `conda create --name test_env`
-----------------------
Install pytorch
```bash
# taken from https://github.com/pytorch/pytorch#from-source
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

# run compilation
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
```

Install detectron2
```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Install torchvision
```bash
git clone https://github.com/pytorch/vision.git
cd vision
python setup.py install
```

### Install pip dependencies for Detectron2
```bash
python -m pip install opencv-python
python -m pip install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
```
### Get datasets from CVAT

## Tests



### Deploy Nuclio Containers for inference in CVAT
run from $HOME:
nuctl deploy --project-name cvat --path "detectron2/nuclio" --platform local
nuctl deploy --project-name cvat --path "detectron2/ball_gpu_model" --platform local

Weird Behavior encountered:
access was not possible for downloading the weights, building the image still worked. After fixing the access rights, building the nuclio image had now affect. Only after changing some python code the image building was triggered and nuclio attempted to download the weights again.





## Nuclio Stuff
TODO explain



---------------
Install torch with gpu with conda
https://github.com/pytorch/pytorch/issues/10234
https://github.com/pytorch/vision
https://detectron2.readthedocs.io/en/latest/tutorials/install.html#build-detectron2-from-source
https://anaconda.org/conda-forge/detectron2
https://github.com/pytorch/pytorch#from-source

base env is in /home/benji/anaconda3