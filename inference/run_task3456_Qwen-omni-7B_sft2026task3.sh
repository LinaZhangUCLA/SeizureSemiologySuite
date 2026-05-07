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
python task34567_omni_7bsft2task3.py \
    --gpu 3\
    --videos_range 1-1000 \
    --output_dir /home/lina/ssb/SeizureSemiologyBench/inference_result20260327300 \
    --model_name /mnt/SSD1/lina/sft/seizure_omni_sft300 \
    --dataset_dir /mnt/SSD1/ssbenchtest \
    --cache_dir /mnt/SSD1/lina/cache
echo "Done!"
