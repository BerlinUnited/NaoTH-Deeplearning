#! /bin/bash

mlflow_experiment="Ball Classifier CNN YUV422 Y-Only Legacy Patches Border 0 16x16 Combined"
mlflow_server="https://mlflow.berlin-united.com/" # https://mlflow2.berlin-united.com/
mlflow_fail_on_timeout="True"
input_shape="16 16 1"
epochs=2000
batch_size=128
data_root="../../data"
data_train="classification_patches_yuv422_y_only_pil_legacy_border0/classification_patches_yuv422_y_only_pil_legacy_border0_combined_16x16_train_ball_no_ball_X_y.h5"
data_val="classification_patches_yuv422_y_only_pil_legacy_border0/classification_patches_yuv422_y_only_pil_legacy_border0_combined_16x16_validation_ball_no_ball_X_y.h5"
data_test="classification_patches_yuv422_y_only_pil_legacy_border0/classification_patches_yuv422_y_only_pil_legacy_border0_combined_16x16_test_ball_no_ball_X_y.h5"

./train.sh -e "$mlflow_experiment" \
  -s "$mlflow_server" \
  -f "$mlflow_fail_on_timeout" \
  -i "$input_shape" \
  -n "$epochs" \
  -b "$batch_size" \
  -d "$data_root" \
  -t "$data_train" \
  -v "$data_val" \
  -x "$data_test"