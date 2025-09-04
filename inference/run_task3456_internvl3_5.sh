#!/bin/bash

# Activate virtual environment
eval "$(conda shell.bash hook)"
conda activate internvl3_5

# Check if activation was successful
if [ $? -ne 0 ]; then
  echo "Failed to activate conda environment internvl3_5. Exiting."
  exit 1
fi

echo "Conda environment internvl3 activated."

# Run the Python script with specified arguments
python /mnt/SSD3/xinyi/benchmark/task3456_internvl3_5.py \
  --gpu 0 \
  --model_name OpenGVLab/InternVL3_5-8B \
  --dataset_dir /mnt/SSD3/lina/ucla2 \
  --cache_dir /mnt/SSD3/xinyi/benchmark/model_cache \
  --output_dir /mnt/SSD3/xinyi/benchmark/output \
  --videos_range 2315-2316 \
  --tp 1

echo "Done!"
