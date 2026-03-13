# Train a ball model for the Nao

Before: Installastion

```
uv sync
uv pip install tensorflow==2.15.1
```

## All for Top

```
#Download
uv run download_images.py -c TOP
uv run download_images_no_annotation.py -c TOP
#Convert for yolo
uv run convert_annotations_json_to_yolo.py -c TOP
#split dataset
uv run create_train_validation_sets.py -c TOP
#train yolo
uv run train_yolo.py -c TOP
#annotate with yolo
uv run run_model_yolo.py -c TOP
#convert yolo labels back to annotation for further training
uv run convert_back_to_json.py -c TOP

```

## All for Bottom

```
#Download
uv run download_images.py -c BOTTOM
uv run download_images_no_annotation.py -c BOTTOM
#Convert for yolo
uv run convert_annotations_json_to_yolo.py -c BOTTOM
#split dataset
uv run create_train_validation_sets.py -c BOTTOM
#train yolo
uv run train_yolo.py -c BOTTOM
#annotate with yolo
uv run run_model_yolo.py -c BOTTOM
#convert yolo labels back to annotation for further training
uv run convert_back_to_json.py -c BOTTOM

```

## Step by Step

uv sync

1.) _Download annotated images_ and annotations from labelstudio.

```
uv run download_images.py -c TOP
```

```
uv run download_images.py -c BOTTOM
```

2.) _Download not annotated images_, you want to annotate:
download annotated images and annotations from labelstudio.

```
uv run download_images_no_annotation.py -c TOP
```

```
uv run download_images_no_annotation.py -c BOTTOM
```

3.) _Convert .json to yolo.txt_

```
uv run convert_annotations_json_to_yolo.py -c TOP
```

```
uv run convert_annotations_json_to_yolo.py -c BOTTOM
```

Results are saved in \labels.

4.) Split data in train and validation set, using:

```
uv run create_train_validation_sets.py -c TOP
```

```
uv run create_train_validation_sets.py -c BOTTOM
```

This will split with 80/20 (Train/Validation) and still keeps the complete dataset in \all. So you can rerun this to generate different splitttings.

5.) _Train yolo model_ on annotated images

```
uv run train_yolo.py -c TOP
```

```
uv run train_yolo.py -c BOTTOM
```

The results will be saved in
{your_git_folder}\runs\detect\yolo_runs\train

naoth-deeplearning\runs\detect\yolo_runs\train

In runs without github_folder, this path will be different. Check this and save for next step.

You should see a visualisation of the images including the lables as well as the prediction from yolo.

6.) _Run yolo model_ on not annotated images

This will generate new annotated images and annotations. The results will be saved in
naoth-deeplearning\patch_based_training\data\yolo\{bottom\top}

If running without git-folder, the path need to be set with -m

Visuale inspection can be done with the images including bounding boxes and confidence level.

```
uv run run_model_yolo.py -c TOP -m path
```

```
uv run run_model_yolo.py -c BOTTOM  -m path
```

7.) _Convert yolo output_(.txt) into .json

```
uv run convert_back_to_json.py -c TOP
```

```
uv run convert_back_to_json.py -c BOTTOM
```

After this you have generated new annotated data you can use to train the ball_detector.

------------------------------------------
Go into Classifier

uv venv .train
source .train/bin/activate
uv pip install -r requirement_train.txt

1.) Split data
```
```
2.) Train model 
```
```
3.) Test model in Log - Simulator
- CNNBallDetector --> set classifier and classifierClose
- adapt cnn.threshold and cnn.thresholdClose 

