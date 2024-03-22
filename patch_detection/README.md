## Datasets

The patch detection uses 3 distinct types of datasets:

- unlabelled patches (balls, balls)
- labelled patches (balls, not balls)
- labelled patches with ball center and radius (balls, not balls)

See datasets.py for detailed documentation of the number of samples, structure of the  
datasets and an easy way to load the datasets from local storage. You can find all  
datasets on the hu storage: `/vol/repl261-vol4/naoth/datasets`.

To run all examples in the patch_detection module, download the following datasets:

- tk03_natural_detection.pkl
- patches_21_23.h5
- classification_ball_no_ball.h5
- classification_ball_no_ball_top_bottom.h5

and copy them to `patch_detection/data/`.
