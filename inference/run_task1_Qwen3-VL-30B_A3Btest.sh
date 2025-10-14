#!/bin/bash


# Activate virtual environment
eval "$(conda shell.bash hook)"
conda activate qwen3vl_moe

# Check if activation was successful
if [ $? -ne 0 ]; then
  echo "Failed to activate conda environment qwenvl. Exiting."
  exit 1
fi

echo "Conda environment qwenvl activated."


# model_name options
# Qwen/Qwen3-VL-30B-A3B-Instruct  2GPU (A100, 80GB)

# video_range 1-2314  eg.1-1000, 1001-2000, 2001-2314
# Run the inference script
python task1_Qwen3-VL-30B-A3B-Instruct.py \
     --gpu 0,1 \
     --videos_range 1-600 \
     --output_dir /home/lina/SeizureSemiologyBench/output \
     --model_name Qwen/Qwen3-VL-30B-A3B-Instruct \
     --dataset_dir /mnt/SSD3/tengyou/seizure_videos/segments/all_dataset \
     --cache_dir /mnt/SSD3/lina/SeizureSemiologyBench/cache
echo "Done!"
