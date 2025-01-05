# Create Yolo v11 Model on Nao Images
We provide a docker image and scripts for training a ultralytics YOLOv11 model on our full images. The images are are downloaded from logs.naoth.de directly. To get the list of image URL's and the annotations the [VAT Database](https://github.com/efcy/visual_analytics) is used.

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
You need to create an .env file first with the environment variables that the scripts expect. Most notably the `REPL_ROOT` variable. This must be set to the folder where you mounted `/vol/repl261-vol4/naoth` from a gruenau server.

If you run on your own system you can run it like this:
```bash
docker compose up -d
docker compose exec yolo_image bash
```

!!! If you are using this on servers owned by the team please be conscious of others using the servers too. For this specify the cpu cores you are going to use. As a rule please leave half of the cores for others. Also coordinate with other people training models.
```bash
services:
  yolo_image:
    cpuset: "4-14"  # equivalent to --cpuset-cpus="4-14"
```

After you are done please stop the container
```bash
docker compose down
```

## Run inference
```
python run_model_in_ls.py -m <model name> -p <project id> <project id> <project id>
```
Have a look at [https://models.naoth.de/](https://models.naoth.de/) for a list of available models. For example you can choose `2024-04-06-yolov8s-best-top.pt`
If the model file is not present in the current working dir it will be downloaded.

## Download or create datasets
You need to download the datasets before you can use them in training.
```
python download_datasets.py -ds <dataset name>
```

### Create new datasets
If you want to create datasets and upload them then you need to make sure you have mounted the correct repl folder with sshfs as mentioned above. This is a bit annoying to do inside the docker container because it does not have the systems variables. But you could set them there manually as well. We recommend only to create new datasets after more data was labeled, otherwise downloading the existing ones are recommended.
```
python create_yolo_datasets.py -c {bottom,top}
```

## Run training
```
python train.py -ds <dataset name> -m <basemodel> -c {bottom,top} -u <your name>
```
An example python call is `python train.py -ds yolo-full-size-detection_dataset_top_2024-04-10.yaml -m yolov8s -c bottom -u "Stella Alice"`

The model argument should either be yolov8n or yolov8


## Build Custom YOLO Image
The pipeline already builds the image. You can pull that with
```bash
docker pull scm.cms.hu-berlin.de:4567/berlinunited/tools/naoth-deeplearning/yolo_image:latest
```
or you can built it locally with
```bash
docker build -t yolo_image:latest .
```
