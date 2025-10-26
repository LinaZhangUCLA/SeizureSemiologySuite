#!/bin/bash


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
# lingshu-medical-mllm/Lingshu-32B  2GPU


# video_range 1-2314  eg.1-1000, 1001-2000, 2001-2314
# Run the inference script
python ../task34567_Lingshu-32B.py \
    --gpu 4,5 \
    --videos_range 1201-1800 \
    --output_dir /home/hubing/SeizureSemiologyBench/output \
    --model_name lingshu-medical-mllm/Lingshu-32B \
    --dataset_dir /mnt/SSD3/lina/ucla2/ssbench \
    --cache_dir /home/hubing/SeizureSemiologyBench/cache 
echo "Done!"
