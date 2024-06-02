# Using Semantic Segmentation for Keypoint Detection.
This is inspired by B-Human (TODO link to paper)

## TODO's
support dataset creation for multiple image downscales
support dataset creation for multiple grid sizes
encode parameters in dataset output paths
create masks better and not during download, originally save full sized masks
implement B-human like ball masks
create more network variants with fewer parameters
implement segmentation network for top as well
implement upload to datasets.naoth.de

## Create datasets
It will download all the labeled images that are available and only if the contain annotation. It will not be downloaded twice. So it's fast running the command again with different options for exporting.
```bash
python create_dataset.py -t y -c top
python create_dataset.py -t y -c bottom
```

## Export model to tflite
```bash
python convert_tflite.py -m models/segmentation_y.keras -ds datasets/validation_ds_y.h5
```
