#!/bin/bash

# This is a simple example script to show how to set up the input parameters 
# for the inference script for SeizureSemiologyBench Task 1

# python /home/hubing/SeizureSemiologyBench/inference/task1_Qwen-2.5-VL-7B-Instruct.py \
# --dataset_dir /home/hubing/ucla/first_half_videos \

python /mnt/SSD3/tengyou/SeizureSemiologyBench/inference/task1_Qwen-2.5-VL-7B-Instruct.py \
    --gpu 2,3 \
    --model_name Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_dir /mnt/SSD3/tengyou/seizure_videos/segments/all_dataset \
    --cache_dir /mnt/SSD3/tengyou/model_cache \
    --output_dir /mnt/SSD3/tengyou/inference \
    --videos_range 0,1 \
    --disable_logs False \

echo "Done!"

# ======================================= Notice =======================================
# 1. If no gpu is specified, the script will use cuda:0
# 2. If no model name is specified, the script will use Qwen/Qwen2.5-VL-7B-Instruct
# 3. No default dataset directory is specified. You need to specify the dataset directory.
# 4. If no output directory is specified, the script will create a "output" folder at the current directory.
# 5. If no cache directory is specified, the script will create a "model_cache" folder at the current directory.
# 6. If no videos range is specified, the script will run the first 1 videos.
# 7. If no disable_logs is specified, the script will print the logs.

# You can directly run the python script after setting the parameters as flags.
# You can also change the parameters in the script and run it.
# ======================================= END =======================================