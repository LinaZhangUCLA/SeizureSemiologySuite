#!/bin/bash



# Activate virtual environment
eval "$(conda shell.bash hook)"
# conda activate qwenvl
conda activate omni3

# Check if activation was successful
if [ $? -ne 0 ]; then
  echo "Failed to activate conda environment qwenvl. Exiting."
  exit 1
fi

echo "Conda environment omni activated."


# model_name options
# Qwen/Qwen2.5-VL-7B-Instruct   1GPU  
# Qwen/Qwen2.5-VL-32B-Instruct  2GPU
# Qwen/Qwen2.5-VL-72B-Instruct  4GPU

# video_range 1-2314  eg.1-1000, 1001-2000, 2001-2314

# Run the inference script
python task12_qwen_3_omni_30b.py \
    --gpu 0,3 \
    --videos_range 1-2s \
    --output_dir /home/lina/SeizureSemiologyBench/t15 \
    --model_name Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --dataset_dir /mnt/SSD3/lina/ucla2/ssbench/task1256_segment_60s \
    --cache_dir /mnt/SSD3/lina/SeizureSemiologyBench/cache 

echo "Done!"
