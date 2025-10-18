#!/bin/bash

# Activate virtual environment
eval "$(conda shell.bash hook)"
conda activate qwenvl

# Check if activation was successful
if [ $? -ne 0 ]; then
  echo "Failed to activate conda environment internvl3_5. Exiting."
  exit 1
fi

echo "Conda environment qwenvl activated."

# Run the Python script with specified arguments
#python /mnt/SSD3/xinyi/benchmark/task3456_internvl3_5.py \
python task34567_Qwen-25-VL.py \
  --gpu 0 \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset_dir /mnt/SSD3/lina/ucla2/ssbench \
  --cache_dir /mnt/SSD3/lina/SeizureSemiologyBench/cache \
  --output_dir /home/lina/SeizureSemiologyBench/qwen10178 \
  --videos_range 1-2 \


echo "Done!"
