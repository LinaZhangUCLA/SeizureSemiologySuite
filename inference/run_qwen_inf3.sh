#!/bin/bash

# Simple script to run Qwen inference
# Usage: ./run_qwen_inf.sh [gpu_id] [num_videos]

# Set GPU (default: 3)
GPU=${1:-7}

# Set number of videos (default: 10)
NUM_VIDEOS=${2:-10}

echo "Running Qwen inference on GPU $GPU with $NUM_VIDEOS videos..."

# Activate virtual environment
eval "$(conda shell.bash hook)"
conda activate qwenvl

# Check if activation was successful
if [ $? -ne 0 ]; then
  echo "Failed to activate conda environment qwenvl. Exiting."
  exit 1
fi

echo "Conda environment qwenvl activated."


# model_name options
# Qwen/Qwen2.5-VL-7B-Instruct   1GPU  
# Qwen/Qwen2.5-VL-32B-Instruct  2GPU
# Qwen/Qwen2.5-VL-72B-Instruct  4GPU


# Run the inference script
python /home/hubing/SeizureSemiologyBench/inference/task1_Qwen-2.5-VL-7B-Instruct.py \
    --gpu 2,3 \
    --max_videos -1 \
    --max_frames 60 \
    --fps 2 \
    --max_new_tokens 2048 \
    --max_retries 10 \
    --output_dir /home/hubing/SeizureSemiologyBench/output\
    --model_name Qwen/Qwen2.5-VL-32B-Instruct \
    --dataset_dir /home/hubing/ucla/all_videos \
    --cache_dir /home/hubing/SeizureSemiologyBench/cache \

echo "Done!"
