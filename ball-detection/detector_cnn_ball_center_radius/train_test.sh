#! /bin/bash

mlflow_experiment="CNN DETECTOR TEST"
input_shape="16 16 1"
epochs=10
batch_size=12880
data_train="ball_center_radius_patches_yuv422_y_only_pil_legacy_border0/ball_center_radius_patches_yuv422_y_only_pil_legacy_border0_combined_16x16_train_ball_center_radius_X_y.h5"
data_val="ball_center_radius_patches_yuv422_y_only_pil_legacy_border0/ball_center_radius_patches_yuv422_y_only_pil_legacy_border0_combined_16x16_validation_ball_center_radius_X_y.h5"


python train.py \
    --mlflow_experiment "$mlflow_experiment" \
    --epochs $epochs \
    --input_shape $input_shape \
    --data_train $data_train \
    --data_val $data_val \
    --batch_size $batch_size \
    --filters 8 8 16 16 \
    --n_dense 16 \
    --regularize True \
    --rescale True \
    --subtract_mean True  \
    --augment_training True \
