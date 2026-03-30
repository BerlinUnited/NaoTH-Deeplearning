# NaoTH Deep Learning - German Open 2026 Edition

---

```
autolabeling/
├── README.md
├── ball_trainings_pipeline.py
├── calculate_labeled_images.py
├── old_evaluate.py
├── pyproject.toml
├── run_model_yolo.py
├── runs/
├── tools.py
├── uv.lock
└── yolo26n.pt
```

---

## Setup

Navigate to the project folder autolabeling:

```bash
cd balldetection2026/autolabeling
```

---

## Workflow - YOLO Pipeline

### Step 1 — Train YOLO Model

Train on the human-annotated images :

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

### Step 2 — Run YOLO Model on Unannotated Images

Runs the trained model on images that have not yet been human-proofed, generating new bounding boxes and annotations:

The name of the current trained model (folder*name, e.g. yolo*{camera}_run_{current_time}) need to be set with -m. If you don't know, just run once without -m and select the model you want, shown in the terminal.

```

uv run run_model_yolo.py -c TOP/BOTTOM

```

Visual inspection can be done with the images, including bounding boxes and confidence level.
