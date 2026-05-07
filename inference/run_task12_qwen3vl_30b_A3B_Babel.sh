#!/bin/bash
#SBATCH --job-name=qwen3vl              # Name of the job
#SBATCH --partition=general             # Partition (queue) to submit to
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks (processes)
#SBATCH --cpus-per-task=32              # Number of CPU cores per task
#SBATCH --mem=500G                      # Total memory (RAM) for the job
#SBATCH --gres=gpu:L40S:8               # Request L40S GPUs
#SBATCH --time=03:00:00                 # Time limit 
#SBATCH --output=/home/prateiks/seizure_local/logs/qwen3vl_32b_%j.out   # Standard output log (%j = job ID)
#SBATCH --error=/home/prateiks/seizure_local/logs/qwen3vl_32b_%j.err    # Standard error log (%j = job ID)

# Create log directory if it doesn't exist
mkdir -p /home/prateiks/seizure_local/logs

# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# Activate virtual environment
eval "$(conda shell.bash hook)"
conda activate qwen3vl_moe_2

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate /mnt/SSD3/lina/my_conda_env/qwen3vl_moe
# which python
# python -c 'import sys; print(sys.executable)'


# Check if activation was successful
if [ $? -ne 0 ]; then
  echo "Failed to activate conda environment. Exiting."
  exit 1
fi

echo "Conda environment qwen3vl_moe activated."

# model_name options
# Qwen/Qwen2.5-VL-7B-Instruct   1GPU  
# Qwen/Qwen2.5-VL-32B-Instruct  2GPU
# Qwen/Qwen2.5-VL-72B-Instruct  4GPU

# video_range 1-2314  eg.1-1000, 1001-2000, 2001-2314

# Run the inference script
python /home/prateiks/SeizureSemiologyBench/inference3/task12_Qwen3-VL-30B-A3B-Instruct.py \
    --gpu 0,1,2,3,4,5,6,7 \
    --videos_range 1-2 \
    --output_dir /home/prateiks/seizure_local/output \
    --model_name Qwen/Qwen3-VL-30B-A3B-Instruct \
    --dataset_dir /home/prateiks/seizure_local/videos/task1256_segment_60s \
    --cache_dir /data/user_data/prateiks 

echo "Done!"
echo "Job finished at: $(date)"
