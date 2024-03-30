#!/bin/bash

latent_dims=(64 32 16 8 4)
models=("vaegan" "vae")
filters="32, 64, 128, 256"

  for latent_dim in "${latent_dims[@]}"; do
            for model in "${models[@]}"; do
                # Execute the Python script with the current combination of arguments
                echo "Running with latent_dim=$latent_dim, model=$model, filters=$filters"
                python train_autoencoder.py --latent_dim $latent_dim --model $model --filters "($filters)" --model_name "filter_32_64_128_256" --epochs 1000
            done
done