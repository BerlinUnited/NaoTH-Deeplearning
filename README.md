# NaoTH-DeepLearning
This repo contains tools for datasets, learning, compiling and evaluating neural networks. This is for all deep learning
done as part of the RoboCup SPL efforts of the Berlin United Team.

## Setup
The scripts were developed with python 3.9. They might work with other python versions as well but they were not tested. You may create a suitable environment with:
```bash
sudo apt install python3.9 python3.9-venv python3.9-dev
python3.9 -m venv venv
source venv/bin/activate
python -m pip install pip -U
python -m pip install -r requirements.txt
```
The dev package is needed for building cppyy.

You need to put the path to the naoth main repo in the variable NAOTH_REPO. You can put the line
```bash
export NAOTH_REPO=/home/stella/RoboCup/Repositories/naoth-2020
```
into your .bashrc for example.

Additionally you have to call the `compile_linux_native.sh` script inside the `$NAOTH_REPO/NaoTHSoccer/Make` folder to compile the naoth lib for your local linux system.

---
In the root folder run `cp config-template.toml config.toml` and adjust the paths inside the newly created `config.toml`file.
The values there are used in most of the datasets scripts.

If you work locally you might want to mount the log folder via sshfs:
```bash
sudo sshfs -o allow_other,uid=1000,gid=1000,ServerAliveInterval=15,ServerAliveCountMax=3,reconnect,IdentityFile=<absolute path to your key> naoth@gruenau4.informatik.hu-berlin.de:/vol/repl261-vol4/naoth/logs/ /mnt/repl
```
set uid and gid to your local user otherwise it will be mounted as root which might not be desireable.

Access only works if your key is authorized to be used with the naoth user. Ask the teamleads if you want your key to be added as well.
Note: if the connection can't be established, try one of the other gruenau servers. For this usecase it does not matter which one.

## Dataset Tools
This folder contains scripts for 
- uploading/downloading datasets to our cvat instance
- creating ball/robots detections datasets that can be later used to train networks running on the nao
- creating datasets that can be used for auto annotation inside our cvat instance


## Auto Annotation
- code for training models used for auto annotation
- document how to set up auto annotation for cvat

## Ball Detection
- networks for ball patch classification, detection and segmentation
- evaluation for those models

## Robot Detection
- maybe combine this with ball detection

## Compiling Neural Networks
- nao devils compilers
- iree
- frugally