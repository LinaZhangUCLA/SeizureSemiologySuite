#!/bin/bash



# Activate virtual environment
# eval "$(conda shell.bash hook)"
# conda activate qwenvl
# conda activate qwen3vl_moe


eval "$(conda shell.bash hook)"
conda activate qwen3vl_moe_2

# Check if activation was successful
if [ $? -ne 0 ]; then
  echo "Failed to activate conda environment qwenvl. Exiting."
  exit 1
fi

echo "Conda environment qwen3vl_moe activated."


# model_name options
# Qwen/Qwen2.5-VL-7B-Instruct   1GPU  
# Qwen/Qwen2.5-VL-32B-Instruct  2GPU
# Qwen/Qwen2.5-VL-72B-Instruct  4GPU

# video_range 1-2314  eg.1-1000, 1001-2000, 2001-2314

# Run the inference script
python /home/ubuntu/SeizureSemiologyBench/inference3/task34567_Qwen-30-VL_moe.py\
    --gpu 0,1,2,3,4,5,6,7 \
    --videos_range 1-2 \
    --output_dir /home/ubuntu/seizure_local/output \
    --model_name Qwen/Qwen3-VL-235B-A22B-Instruct\
    --dataset_dir /home/ubuntu/seizure_local/videos \
    --cache_dir /home/ubuntu/cache

echo "Done!"
