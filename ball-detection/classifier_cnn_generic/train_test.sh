#! /bin/bash

mlflow_experiment="MAX DL TEST"
input_shape="16 16 1"
epochs=1
batch_size=12880
data_train="classification_patches_yuv422_y_only_combined_16x16_train_ball_no_ball_X_y.h5"
data_val="classification_patches_yuv422_y_only_combined_16x16_validation_ball_no_ball_X_y.h5"
data_test="classification_patches_yuv422_y_only_combined_16x16_test_ball_no_ball_X_y.h5"

augment_training_values=("True" "False")
regularize_values=("True" "False")
filters_values=("4 4 4 4" "4 4 8 8" "8 8 16 16" "8 16 32 64" "16 32 64 128")
n_dense_values=("16" "32" "64" "128")


# Loop over all combinations of parameter values
for augment_training in "${augment_training_values[@]}"; do
  for regularize in "${regularize_values[@]}"; do
    for filters in "${filters_values[@]}"; do
      for n_dense in "${n_dense_values[@]}"; do
        echo ""
        echo "Running with augment_training=$augment_training, filters=$filters, regularize=$regularize, dropout=$dropout, n_dense=$n_dense"
        echo ""
        python train.py \
            --mlflow_experiment "$mlflow_experiment" \
            --epochs $epochs \
            --input_shape $input_shape \
            --augment_training $augment_training \
            --filters $filters \
            --regularize $regularize \
            --n_dense $n_dense \
            --data_train $data_train \
            --data_val $data_val \
            --data_test $data_test \
            --batch_size $batch_size \
            --rescale True \
            --subtract_mean True 
      done
    done
  done
done
