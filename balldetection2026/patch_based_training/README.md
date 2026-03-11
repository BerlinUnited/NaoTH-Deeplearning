# Train a ball model for the Nao
Before: Installastion
```
uv sync
uv pip install tensorflow==2.15.1 
```

1.) *Download annotated images* and annotations from labelstudio.
```
uv run download_images.py -c TOP
```
```
uv run download_images.py -c BOTTOM
```
2.) *Download not annotated images*, you want to annotate: 
download annotated images and annotations from labelstudio.
```
uv run download_images_no_annotation.py -c TOP
```
```
uv run download_images_no_annotation.py -c BOTTOM 
```

3.) *Convert .json to yolo.txt*
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

5.) *Train yolo model* on annotated images
```
uv run train_yolo.py -c TOP
```
```
uv run train_yolo.py -c BOTTOM 
```
The results will be saved in 
{your_git_folder}\runs\detect\yolo_runs\train
naoth-deeplearning\runs\detect\yolo_runs\train

You should see a visualisation of the images including the lables as well as the prediction from yolo. 

6.) *Run yolo model* on not annotated images

This will generate new annotated images and annotations. The results will be saved in 
naoth-deeplearning\patch_based_training\data\yolo\{bottom\top}

Visuale inspection can be done with the images including bounding boxes and confidence level. 

```
uv run run_model_yolo.py -c TOP
```
```
uv run run_model_yolo.py -c BOTTOM 
```

7.) *Convert yolo output*(.txt) into .json

```
uv run convert_back_to_json.py -c TOP
```
```
uv run convert_back_to_json.py -c BOTTOM 
```


After this you have generated new annotated data you can use to train the ball_detector. 