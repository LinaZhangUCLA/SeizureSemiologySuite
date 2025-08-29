# Simple script to run InternVL3.5 inference
# Usage: ./run_internvl3_5_inf.sh [gpu_id] [num_videos] [tp]
#   gpu_id    : CUDA device index (default: 0)
#   tp        : tensor parallel size (default: 1)
#   num_videos: how many videos to process; -1 = all (default: -1)


# --------- Args with defaults ---------
GPU=${1:-7}
TP=${2:-1}
NUM_VIDEOS=${3:--1}

echo "Running InternVL3.5 inference on GPU ${GPU} | max_videos=${NUM_VIDEOS} | tp=${TP}"

# --------- Activate conda env ---------
eval "$(conda shell.bash hook)"
conda activate internvl3
if [ $? -ne 0 ]; then
  echo "Failed to activate conda environment 'internvl3'. Exiting."
  exit 1
fi
echo "Conda environment 'internvl3' activated."

# --------- Run inference ---------
python task1_internvl3.5_8B.py \
  --gpu 1 \
  --tp 1 \
  --model_name OpenGVLab/InternVL3_5-8B \
  --dataset_dir seizure_videos/segments/all_dataset/ \
  --output_dir . \
  --cache_dir ./model_cache/

echo "Done!"




