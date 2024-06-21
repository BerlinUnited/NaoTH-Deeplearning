#! /bin/bash

mlflow_experiment="CNN CLASSIFIER TEST"
input_shape="16 16 1"
epochs=10
batch_size=12880
data_train="classification_patches_yuv422_y_only_pil_legacy_border0/classification_patches_yuv422_y_only_pil_legacy_border0_combined_16x16_train_ball_no_ball_X_y.h5"
data_val="classification_patches_yuv422_y_only_pil_legacy_border0/classification_patches_yuv422_y_only_pil_legacy_border0_combined_16x16_train_ball_no_ball_X_y.h5"
data_test="classification_patches_yuv422_y_only_pil_legacy_border0/classification_patches_yuv422_y_only_pil_legacy_border0_combined_16x16_train_ball_no_ball_X_y.h5"


python train.py \
    --mlflow_experiment "$mlflow_experiment" \
    --epochs $epochs \
    --input_shape $input_shape \
    --data_train $data_train \
    --data_val $data_val \
    --data_test $data_test \
    --batch_size $batch_size \
    --augment_training True \
    --filters 8 8 16 16 \
    --regularize True \
    --n_dense 16 \
    --rescale True \
    --subtract_mean True 

