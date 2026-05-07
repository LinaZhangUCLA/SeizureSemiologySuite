#!/bin/bash



# Activate virtual environment
eval "$(conda shell.bash hook)"
# conda activate qwenvl
conda activate internvl3_5

# Check if activation was successful
if [ $? -ne 0 ]; then
  echo "Failed to activate conda environment qwenvl. Exiting."
  exit 1
fi

echo "Conda environment internvl3_5 activated."


# model_name options
# Qwen/Qwen2.5-VL-7B-Instruct   1GPU  
# Qwen/Qwen2.5-VL-32B-Instruct  2GPU
# Qwen/Qwen2.5-VL-72B-Instruct  4GPU

# video_range 1-2314  eg.1-1000, 1001-2000, 2001-2314

# Run the inference script
python task1_internvl3.5_8B.py \
    --gpu 6 \
    --tp 1 \
    --videos_range 1801-2100 \
    --output_dir /home/hubing/SeizureSemiologyBench/output \
    --model_name OpenGVLab/InternVL3_5-8B \
    --dataset_dir /home/hubing/SeizureSemiologyBench/ucla  \
    --cache_dir /home/hubing/SeizureSemiologyBench/cache 

echo "Done!"
