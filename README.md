# NaoTH-DeepLearning

## Setup
You need to put the path to the naoth main repo in the variable NAOTH_REPO. You can put the line
```bash
export NAOTH_REPO=/home/stella/RoboCup/Repositories/naoth-2020
```
into your .bashrc for example.

Additionally you have to call the `compile_linux_native.sh` script inside the `NAOTH_REPO/NaoTHSoccer/Make` folder to compile the naoth lib for your local linux system.

This repo contains tools for datasets, learning, compiling and evaluating neural networks. This is for all deep learning
done as part of the RoboCup SPL efforts of the Berlin United Team.

## Dataset Tools
- create pkl files for the old TK3 dataset as classification, detection and segmentation dataset
- create pkl files for bhuman dataset
- download annotated cvat datasets and create patches from those datasets
- augment datasets
- create datasets for large networks used for auto annotation in cvat
- create robot detection datasets

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