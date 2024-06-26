#! /bin/bash

mlflow_experiment="Ball Detector CNN YUV422 Y-Only Legacy Patches Border 0 Combined"
mlflow_server="https://mlflow.berlin-united.com/" # https://mlflow2.berlin-united.com/
mlflow_fail_on_timeout="True"
input_shape="16 16 1"
epochs=2000
batch_size=128
data_root="../../data"
data_train="ball_radius_center_patches_yuv422_y_only_pil_legacy_border0/ball_radius_center_patches_yuv422_y_only_pil_legacy_border0_combined_16x16_train_ball_radius_center_X_y.h5"
data_val="ball_radius_center_patches_yuv422_y_only_pil_legacy_border0/ball_radius_center_patches_yuv422_y_only_pil_legacy_border0_combined_16x16_validation_ball_radius_center_X_y.h5"


./train.sh -e "$mlflow_experiment" \
  -s "$mlflow_server" \
  -f "$mlflow_fail_on_timeout" \
  -i "$input_shape" \
  -n "$epochs" \
  -b "$batch_size" \
  -d "$data_root" \
  -t "$data_train" \
  -v "$data_val" 
