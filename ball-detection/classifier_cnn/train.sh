#! /bin/bash


# Default values
data_root="../../data"
epochs=2000
batch_size=128
mlflow_fail_on_timeout="True"
augment_training_values=("True" "False")
regularize_values=("True" "False")
filters_values=("4 4 4 4" "4 4 8 8" "8 8 16 16" "8 16 32 64")
n_dense_values=("16" "32" "64" "128")

# Get mandatory arguments and override default values where applicable
while getopts e:s:f:i:n:b:d:t:v:x: flag
do
    case "${flag}" in
        e) mlflow_experiment=${OPTARG};;
        s) mlflow_server=${OPTARG};;
        f) mlflow_fail_on_timeout=${OPTARG};;
        i) input_shape=${OPTARG};;
        n) epochs=${OPTARG};;
        b) batch_size=${OPTARG};;
        d) data_root=${OPTARG};;
        t) data_train=${OPTARG};;
        v) data_val=${OPTARG};;
        x) data_test=${OPTARG};;
    esac
done

# Check if data parameters are provided
if [ -z "$data_root" ] || [ -z "$data_train" ] || [ -z "$data_val" ] || [ -z "$data_test" ]; then
    echo "Error: data_root (-d), data_train (-t), data_val (-v), and data_test (-x) parameters are required."
    exit 1
fi

if [ -z "$mlflow_server" ] ; then
    echo "Error: mlflow_server parameter (-s) is required."
    exit 1
fi

if [ -z "$mlflow_experiment" ] ; then
    echo "Error: mlflow_experiment parameter (-e) is required."
    exit 1
fi

if [ -z "$input_shape" ] ; then
    echo "Error: input_shape parameter (-i) is required."
    exit 1
fi


# Display the arguments
echo "mlflow_experiment: $mlflow_experiment"
echo "mlflow_server: $mlflow_server"
echo "mlflow_fail_on_timeout: $mlflow_fail_on_timeout"
echo "input_shape: $input_shape"
echo "epochs: $epochs"
echo "batch_size: $batch_size"
echo "data_root: $data_root"
echo "data_train: $data_train"
echo "data_val: $data_val"
echo "data_test: $data_test"


# Loop over all combinations of parameter values
for augment_training in "${augment_training_values[@]}"; do
  for regularize in "${regularize_values[@]}"; do
    for filters in "${filters_values[@]}"; do
      for n_dense in "${n_dense_values[@]}"; do
        echo ""
        echo "Running with augment_training=$augment_training, filters=$filters, regularize=$regularize, n_dense=$n_dense"
        echo ""
        python train.py \
            --mlflow_experiment "$mlflow_experiment" \
            --mlflow_server "$mlflow_server" \
            --mlflow_fail_on_timeout "$mlflow_fail_on_timeout" \
            --epochs $epochs \
            --input_shape $input_shape \
            --augment_training $augment_training \
            --filters $filters \
            --regularize $regularize \
            --n_dense $n_dense \
            --data_root $data_root \
            --data_train $data_train \
            --data_val $data_val \
            --data_test $data_test \
            --batch_size $batch_size \
            --rescale True \
            --subtract_mean True &
      done
      wait
    done
  done
done
