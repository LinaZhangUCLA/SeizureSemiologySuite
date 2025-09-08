#!/bin/bash


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
# video_range 1-2314  eg.1-1000, 1001-2000, 2001-2314

# Run the inference script
python ../task3456_Qwen-25-VL-7B-Instruct.py \
    --gpu 4,5,6,7 \
    --videos_range 301-600 \
    --output_dir /home/hubing/SeizureSemiologyBench/output \
    --model_name Qwen/Qwen2.5-VL-72B-Instruct \
    --dataset_dir /home/hubing/SeizureSemiologyBench/ucla2 \
    --cache_dir /home/hubing/SeizureSemiologyBench/cache 

echo "Done!"
