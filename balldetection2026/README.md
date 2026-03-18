# NaoTH Deep Learning - German Open 2026 Edition

---

```
balldetection2026/
├── autolabeling/              # YOLO pipeline for auto-generating labels
│   ├── create_training_data_yolo.py   # Step 1
│   ├── train_model_yolo.py            # Step 2
│   ├── run_model_yolo.py              # Step 3
│   ├── data_bottom.yaml
│   └── data_top.yaml
├── classifier_patch_based/    # Patch-based classification model and training scripts
│   ├── create_patches.py
│   ├── create_training_data_classifier.py
│   ├── model.py
│   ├── evaluate_classifier.py
│   └── train.py              # Training with data augmentation + Keras → TFLite export
└── data/                     # All data: images, annotations, and patches
```
---

## Setup
 
Navigate to the project folder and install the environment:
 
```bash
cd balldetection2026
uv sync
```
 
---

## Workflow - YOLO Pipeline
 
### Step 1 — Download Images & Annotations
Download images and annotations from Label Studio, including images not human-proofed:
```
uv run download.py -c TOP/BOTTOM -l 683 -m "annotated"/"not_annotated"/"both"
```
 
**Output paths:**
 
| Content | Path |
|---|---|
| Images | `balldetection2026/data/{camera}/human_proofed/images/` |
| Annotations | `balldetection2026/data/{camera}/human_proofed/{annotations}/` |
| Non-proofed images | `balldetection2026/data/{camera}/not_human_proofed/{annotations}/` |
 
 
> **Note:** Annotations may be empty if a human reviewer clicked *Submit* without marking anything (e.g. when no ball is visible).
 
---
 
### Step 2 — Create YOLO Training Data
 
Splits the dataset into validation and training sets and converts annotations from `.json` to `.txt`:

```
uv run autolabeling/create_training_data_yolo.py -c TOP/BOTTOM"
```

YOLO-format labels are written to `balldetection2026/data/{camera}/human_proofed/labels/`. Both `labels/` and `images/` are split into `val/` and `train/` subdirectories (the unsplit originals are kept in `all/`).
 
---
 
### Step 3 — Train YOLO Model

Train on the human-annotated images:
```
uv run autolabeling/train_model_yolo.py -c TOP/BOTTOM"
```

The results will be saved in
data/{camera}/autolabel_model/yolo_{camera}_run_{current_time}/
 
The output includes visualisations of images with ground-truth labels and YOLO predictions overlaid.


### Step 4 — Run YOLO Model on Unannotated Images
 
Runs the trained model on images that have not yet been human-proofed, generating new bounding boxes and annotations:

The name of the current trained model (folder_name, e.g. yolo_{camera}_run_{current_time}) need to be set with -m. If you dont know, just run once without -m and select the model you want, shown in the terminal. 

```
uv run run_model_yolo.py -c TOP/BOTTOM -m path
```

Visuale inspection can be done with the images including bounding boxes and confidence level.

**Newly generated annotations are written to:**
balldetection2026/data/{camera}/not_human_proofed/annotations (as well as the yolo labels)

These can now be used to train the ball detector.

## Workflow - classifier 

### Prperation Step 
Get the BallCandidatePatches from the log-files. 

```
uv run log_iteration.py -l 679,683
```


### Step 1 Create Patches

### Step 2 Train model

---
 
## Testing the Model in the Log Simulator
 
### 1. Load the TFLite model
 
Place the `.tflite` file in:
```
naoth-2020/NaoTHSoccer/Config/
```
 
### 2. Register the classifier (first-time setup only)
 
If not previously configured, add the classifier path in:
```
naoth-2020/NaoTHSoccer/Source/Cognition/Modules/VisualCortex/BallDetector/CNNBallDetector.cpp
```
 
### 3. Configure the Log Simulator
 
In the Log Simulator, open `CNNBallDetector` and:
 
- Set `classifier` and `classifierClose` to point to your `.tflite` file
- Tune `cnn.threshold` and `cnn.thresholdClose` as needed
 