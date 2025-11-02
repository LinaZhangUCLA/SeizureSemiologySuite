#!/bin/bash


# Activate virtual environment
eval "$(conda shell.bash hook)"
conda activate lingshu

# Check if activation was successful
if [ $? -ne 0 ]; then
  echo "Failed to activate conda environment qwenvl. Exiting."
  exit 1
fi

echo "Conda environment lingshu activated."


# model_name options
# Qwen/Qwen2.5-VL-7B-Instruct   1GPU  
# Qwen/Qwen2.5-VL-32B-Instruct  2GPU
# Qwen/Qwen2.5-VL-72B-Instruct  4GPU

# video_range 1-2314  eg.1-1000, 1001-2000, 2001-2314
# Run the inference script
python ../task12_lingshu_32b.py \
    --gpu 4,5 \
    --videos_range 1201-1800 \
    --model_name lingshu-medical-mllm/Lingshu-32B \
    --output_dir /home/hubing/SeizureSemiologyBench/output \
    --dataset_dir /home/hubing/SeizureSemiologyBench/ucla  \
    --cache_dir /home/hubing/SeizureSemiologyBench/cache 
echo "Done!"
