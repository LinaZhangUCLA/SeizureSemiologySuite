#!/bin/bash
#SBATCH --job-name=finetune     
#SBATCH --partition=general    
#SBATCH --nodes=1       
#SBATCH --ntasks=8      
#SBATCH --cpus-per-task=4  
#SBATCH --mem=96G       
#SBATCH --gres=gpu:L40S:8 
#SBATCH --time=12:00:00    
#SBATCH --output=/home/prateiks/SeizureSemiologyBench/logs/omni_finetune_%j_%t.out

# Activate virtual environment
eval "$(conda shell.bash hook)"
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

cd /home/prateiks/SeizureSemiologyBench/inference3/task1_Qwen25_omni_7B_babel

# Split videos 1-350 across 8 GPUs (roughly 44 videos per GPU)
# Task 0 (GPU 0): 1-44, Task 1 (GPU 1): 45-88, Task 2 (GPU 2): 89-132, Task 3 (GPU 3): 133-176
# Task 4 (GPU 4): 177-220, Task 5 (GPU 5): 221-264, Task 6 (GPU 6): 265-308, Task 7 (GPU 7): 309-350

# Define video ranges for each task (exported for use in srun)
export VIDEO_RANGES="1-44:45-88:89-132:133-176:177-220:221-264:265-308:309-350"

# Use srun to launch all 8 tasks in parallel
# Each task will have $SLURM_PROCID set to 0-7
srun bash -c '
    TASK_ID=$SLURM_PROCID
    GPU_ID=$TASK_ID
    IFS=":" read -ra RANGES <<< "$VIDEO_RANGES"
    VIDEO_RANGE=${RANGES[$TASK_ID]}
    
    echo "Task $TASK_ID: Using GPU $GPU_ID, processing videos $VIDEO_RANGE"
    
    python ../task12_qwen2_5_omni_7b.py \
        --gpu $GPU_ID \
        --videos_range $VIDEO_RANGE \
        --output_dir /home/prateiks/SeizureSemiologyBench/output \
        --model_name CedrusLNZ/seizure_omni_sft \
        --dataset_dir /home/prateiks/data/finetune_test_videos/task1256_segment_30s \
        --cache_dir /home/prateiks/data/hf_cache
    
    echo "Task $TASK_ID done!"
'
