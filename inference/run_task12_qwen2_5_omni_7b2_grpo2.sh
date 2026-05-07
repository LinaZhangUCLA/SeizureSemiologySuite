#!/bin/bash



# Activate virtual environment
eval "$(conda shell.bash hook)"
# conda activate qwenvl
conda activate omni

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
python task12_qwen2_5_omni_7b.py \
    --gpu 2 \
    --videos_range 181-1000 \
    --output_dir /home/lina/ssb/SeizureSemiologyBench/inference_result \
    --model_name CedrusLNZ/seizure_omni_grpo \
    --dataset_dir /mnt/SSD3/ssbenchtest/task1256_segment_30s \
    --cache_dir /mnt/SSD3/lina/SeizureSemiologyBench/cache 

echo "Done!"
