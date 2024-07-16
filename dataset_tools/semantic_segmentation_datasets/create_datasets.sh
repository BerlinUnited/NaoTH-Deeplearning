#! /bin/bash
color_modes=("YUV422_Y_ONLY_PIL" "YUV422_PIL" )
scale_factors=(4 2 8)
ball_only_values=("True" "False")

for color_mode in "${color_modes[@]}"; do
  for scale_factor in "${scale_factors[@]}"; do
    for ball_only in "${ball_only_values[@]}"; do
      #echo "Running with color_mode=$color_mode, patch_type=$patch_type, border=$border"
      python create_datasets.py --color_mode $color_mode --camera bottom --scale_factor $scale_factor --grid 15 20 --ball_only $ball_only
    done
  done
done