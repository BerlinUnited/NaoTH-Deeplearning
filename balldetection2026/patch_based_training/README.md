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
3.) Split data in train and validation set, using: 
```
uv run create_train_validation_sets.py -c TOP
```
```
uv run create_train_validation_sets.py -c BOTTOM
```
This will split with 80/20 (Train/Validation) and still keeps the complete dataset in \all. So you can rerun this to generate different splitttings. 

4.) *Train yolo model* on annotated images
```
uv run train_yolo.py -c TOP
```
```
uv run train_yolo.py -c BOTTOM 
```
The results will be saved in 

5.)