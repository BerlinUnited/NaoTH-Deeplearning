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

A defaut config for training is a .yaml file and looks like this:

```yaml
target_class: "Ball"
modelsize: "n" # n, m, l, x
camera: "TOP" # BOTTOM or TOP
# optional
log_ids:
  - 675
ls_project_ids:
  - 7694
epochs: 1
split_ratio: 0.8 # default is 0.8
# optional, otherwise it is random
seed: 424242 
```

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

A defaut config for training is a .yaml file and looks like this:

```
target_class: "Ball"

camera: "TOP" # BOTTOM or TOP

#log_ids:
#  - 675

ls_project_ids:
  - 12454

# mlflow_run_name: "nervous-slug-942" # choose a specific run from the MLFlow experiment with the name TARGET_CLASS-CAMERA-classifier-model

# model: # you can also use your own model .pt file

# num_images: 1

```
