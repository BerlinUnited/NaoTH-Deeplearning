# NaoTH Deep Learning - German Open 2026 Edition

```
autolabeling/
├── README.md
├── ball_trainings_pipeline.py
├── pyproject.toml
├── run_model_yolo.py
├── tools.py
└── uv.lock
```

---

## Setup

Before using, you must set the following environment variables.
It is recommended to store them locally in an env.sh file for quicker setup in future runs.

```bash
export VAT_API_URL=***
export VAT_API_TOKEN=***
export LABELSTUDIO_API_KEY=***
export MLFLOW_TRACKING_USERNAME=***
export MLFLOW_TRACKING_PASSWORD=***
export MLFLOW_USER=***
```
Then navigate to the project folder 'autolabeling':

```bash
cd balldetection2026/autolabeling
source ../env.sh  #set environment variables
```

---

### Step 1 — Train YOLO Model

Train on the human-annotated images. 
The images are downloaded and maped with the annotations from labelstudio. 

```
uv run ball_trainings_pipeline.py
```

| Options                                 | Example         |
| :-------------------------------------- | :-------------- |
| Camera                                  | -c BOTTOM/TOP   |
| epochs number                           | -e 200 (int)    |
| log ids                                 | -l 001,002      |
| model size                              | -m n/s/m/l/x    |
| split ratio (optional) <br> default 0.8 | -r 0.8 (float)  |
| seed (optional) <br> default random     | -s 753238 (int) |

The results will be saved in

runs/{camera}/{run_timestamp}/autolabel_model/

The output includes visualizations of images with ground-truth labels and YOLO predictions overlaid.

Last line in terminal print modelname and Labelstudio-ID used. 
Needed for next step!

### Step 2 — Run YOLO Model on Unannotated Images


Run the trained model on images that have not yet been human-proofed, 
generating new annotations and directly push to Labelstudio. 

The name of the trained model (folder*name, e.g. yolo*{camera}_run_{current_time}) need to be set with -m. If you don't know, just run once without -m and select the model you want, shown in the terminal.


If you dont want to run on all images, you can set the amount of images with image numbers. 
Those is the only optional argument. 

```
uv run run_model_yolo.py 

```

| Options                                 | flag            |
| :-------------------------------------- | :-------------- |
| Camera                                  | -c BOTTOM/TOP   |
| model                                   | -m              |
| project (labelstudioproject ID)         | -p              |
| num images                              | -n 50 (int)     |
