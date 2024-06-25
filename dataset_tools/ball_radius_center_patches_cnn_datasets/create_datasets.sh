#! /bin/bash

color_modes=("YUV422_Y_ONLY_PIL" "YUV422_PIL" )
patch_types=("patches" "patches_segmentation")
borders=(0 10 20)
patch_sizes=(16 32)

# Loop over each combination of values
for color_mode in "${color_modes[@]}"; do
  for patch_type in "${patch_types[@]}"; do
    for border in "${borders[@]}"; do
      for patch_size in "${patch_sizes[@]}"; do
        echo "Running with color_mode=$color_mode, patch_type=$patch_type, border=$border"
        python create_datasets.py --color_mode $color_mode --patch_type $patch_type --border $border --patch_size $patch_size
      done
    done
  done
done