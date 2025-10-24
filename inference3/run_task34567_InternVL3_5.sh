#!/bin/bash

# Activate virtual environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/SSD3/lina/my_conda_env/internvl3_5
which python
# python -c 'import sys; print(sys.executable)'

# Check if activation was successful
if [ $? -ne 0 ]; then
  echo "Failed to activate conda environment internvl3_5. Exiting."
  exit 1
fi

echo "Conda environment internvl3_5 activated."

# Run the Python script with specified arguments
#python /mnt/SSD3/xinyi/benchmark/task3456_internvl3_5.py \
python task34567_InternVL3_5.py \
  --gpu 0 \
  --model_name OpenGVLab/InternVL3_5-8B-Instruct \
  --dataset_dir /mnt/SSD3/lina/ucla2/ssbench \
  --cache_dir /mnt/SSD4/prateik/cache \
  --output_dir /mnt/SSD1/prateik/SeizureSemiologyBench/output \
  --videos_range 1-2 \


echo "Done!"
