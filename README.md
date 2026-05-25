# NaoTH Deep Learning

https://labelstudio.berlin-united.com/
https://mlflow.berlin-united.com/#/experiments


## Create an auto annotation model
The code for training a model that can automate the previous step is in balldetection2026/autolabeling folder. It requires that some images are already annotated by hand.

Setup python with `uv sync`

To download the trainings data you have to run: `python create_trainings_data.py`. Currently it uses the trainings data for validation as well. Feel free to fix this.

Set the environment variables as described in the slack thread 

Run training with `python train.py -c BOTTOM` or `python train.py -c TOP`  your can see the results in MLFLOW: https://mlflow.berlin-united.com/#/experiments/3/runs/be49053c082e418bb800b3f4765ffba7/model-metrics

username and password is in the slack thread.

TODO: add diff between top and bottom camera
TODO: add train/validation split
TODO: add test/eval stage

TODO: document how to apply this model

## Train model on patches
In the folder balldetection2026/patch_based_training are scripts that can use annotations from labelstudio and create patches (32x32). Thise can be given as trainings input. It would be better to use patches from logs. 
Thomas wanted to look into better patches for the new ball. As soon as he is finished with this we can extract those patches from the logs and use them as trainings input.

For improving training on patches we can use the ball detection from Max that served us really well for the last 2 years: The code can be found in ball-detection/detector_cnn_ball_radius_center
