#!/bin/bash

# Create log directory if it doesn't exist
mkdir -p /home/ubuntu/seizure_local/logs

# Activate virtual environment
eval "$(conda shell.bash hook)"
conda activate qwen3vl_moe_2

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate /mnt/SSD3/lina/my_conda_env/qwen3vl_moe
# which python
# python -c 'import sys; print(sys.executable)'

# Check if activation was successful
if [ $? -ne 0 ]; then
  echo "Failed to activate conda environment. Exiting."
  exit 1
fi

echo "Conda environment qwen3vl_moe activated."

# model_name options
# Qwen/Qwen2.5-VL-7B-Instruct   1GPU  
# Qwen/Qwen2.5-VL-32B-Instruct  2GPU
# Qwen/Qwen2.5-VL-72B-Instruct  4GPU

# video_range 1-2314  eg.1-1000, 1001-2000, 2001-2314

# Run the inference script
python /home/ubuntu/SeizureSemiologyBench/inference3/task12_Qwen3-VL-30B-A3B-Instruct.py \
    --gpu 0,1,2,3,4,5,6,7 \
    --videos_range 1-2 \
    --output_dir /home/ubuntu/seizure_local/output \
    --model_name Qwen/Qwen3-VL-235B-A22B-Instruct \
    --dataset_dir /home/ubuntu/seizure_local/videos/task1256_segment_60s \
    --cache_dir /home/ubuntu/cache

echo "Done!"
# echo "Job finished at: $(date)"
