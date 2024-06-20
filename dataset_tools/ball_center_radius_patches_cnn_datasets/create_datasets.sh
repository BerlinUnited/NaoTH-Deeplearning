#! /bin/bash

color_modes=("YUV422_Y_ONLY_PIL" "YUV422_PIL" )
# patch_types=("patches" "patches_segmentation")
borders=(0 10 20)

# Loop over each combination of values
for color_mode in "${color_modes[@]}"; do
    for border in "${borders[@]}"; do
      echo "Running with color_mode=$color_mode, patch_type=$patch_type, border=$border"
      python create_datasets.py --color_mode $color_mode --patch_type "patches_segmentation" --border $border
    done
  done
done