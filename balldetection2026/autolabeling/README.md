# NaoTH Deep Learning - German Open 2026 Edition

---

```

balldetection2026/

└── autolabeling/ # YOLO pipeline for auto-generating labels

├── ball_trainings_pipeline.py # Step 1

├── run_model_yolo.py # Step 2

├── datasets/

└── autolabel_models/

```

---

## Setup

Navigate to the project folder and install the environment:

```bash

cd balldetection2026/autolabeling

uv sync

```

---

## Workflow - YOLO Pipeline

### Step 1 — Train YOLO Model

Train on the human-annotated images:

```

uv run ball_trainings_pipeline.py

```

| Feature                  | Example    |
| :----------------------- | :--------- |
| Camera (BOTTOM/TOP)      | -c BOTTOM  |
| epochs number            | -e 200     |
| log ids                  | -l 001,002 |
| model size n, s, m, l, x | -s n       |

The results will be saved in

runs/{camera}/{run_timestamp}/autolabel_model/

The output includes visualizations of images with ground-truth labels and YOLO predictions overlaid.

### Step 2 — Run YOLO Model on Unannotated Images

Runs the trained model on images that have not yet been human-proofed, generating new bounding boxes and annotations:

The name of the current trained model (folder*name, e.g. yolo*{camera}_run_{current_time}) need to be set with -m. If you don't know, just run once without -m and select the model you want, shown in the terminal.

```

uv run run_model_yolo.py -c TOP/BOTTOM -m path

```

Visual inspection can be done with the images, including bounding boxes and confidence level.
