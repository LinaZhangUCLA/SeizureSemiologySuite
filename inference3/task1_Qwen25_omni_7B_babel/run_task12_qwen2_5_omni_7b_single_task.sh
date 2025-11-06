#!/bin/bash
#SBATCH --job-name=finetune     
#SBATCH --partition=general    
#SBATCH --nodes=1       
#SBATCH --ntasks=1      
#SBATCH --cpus-per-task=4  
#SBATCH --mem=8G       
#SBATCH --gres=gpu:L40S:1 
#SBATCH --time=01:00:00    
#SBATCH --output=/home/prateiks/SeizureSemiologyBench/logs/omni_finetune_%j.out

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

# Run the inference script
python ../task12_qwen2_5_omni_7b.py \
    --gpu 0 \
    --videos_range 1-5 \
    --output_dir /home/prateiks/SeizureSemiologyBench/output \
    --model_name CedrusLNZ/seizure_omni_sft \
    --dataset_dir /home/prateiks/data/finetune_test_videos/task1256_segment_30s \
    --cache_dir /home/prateiks/data/hf_cache  

echo "Done!"
