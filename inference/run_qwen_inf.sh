#!/bin/bash

# Simple script to run Qwen inference
# Usage: ./run_qwen_inf.sh [gpu_id] [num_videos]

# Set GPU (default: 3)
GPU=${1:-3}

# Set number of videos (default: 10)
NUM_VIDEOS=${2:-5}

echo "Running Qwen inference on GPU $GPU with $NUM_VIDEOS videos..."

# Activate virtual environment
source .venv/bin/activate

# Run the inference script
python /mnt/SSD3/tengyou/SeizureVLM/evaluation/ExtractFeature_qwen-2.5-vl-new.py \
    --gpu $GPU \
    --max_videos $NUM_VIDEOS \
    --max_frames 30 \
    --fps 1 \
    --max_new_tokens 2048 \
    --max_retries 10 \
    --output_dir /mnt/SSD3/tengyou/inference\
    --model_name Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_dir /mnt/SSD3/tengyou/seizure_videos/segments/all_dataset/ \

echo "Done!"
