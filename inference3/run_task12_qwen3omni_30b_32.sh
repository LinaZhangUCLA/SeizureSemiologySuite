#!/bin/bash



# Activate virtual environment
eval "$(conda shell.bash hook)"
conda activate omni3

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate /mnt/SSD3/lina/my_conda_env/qwen3vl_moe
# which python
# python -c 'import sys; print(sys.executable)'


# Check if activation was successful
if [ $? -ne 0 ]; then
  echo "Failed to activate conda environment qwenvl. Exiting."
  exit 1
fi

echo "Conda environment omni3 activated."


# model_name options
# Qwen/Qwen2.5-VL-7B-Instruct   1GPU  
# Qwen/Qwen2.5-VL-32B-Instruct  2GPU
# Qwen/Qwen2.5-VL-72B-Instruct  4GPU

# video_range 1-2314  eg.1-1000, 1001-2000, 2001-2314

# Run the inference script
python task12_qwen_3_omni_30b2.py \
    --gpu 1,2,3 \
    --videos_range 1-300 \
    --output_dir /home/lina/ssb/SeizureSemiologyBench/inference_result \
    --model_name Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --dataset_dir /mnt/SSD3/lina/ucla2/ssbench/task1256_segment_60s \
    --cache_dir /mnt/SSD3/lina/SeizureSemiologyBench/cache

echo "Done!"
