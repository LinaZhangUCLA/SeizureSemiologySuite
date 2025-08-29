conda activate internvl3_5

huggingface-cli login  # When prompted, paste your token (e.g., hf_xxx...) and press Enter to confirm.

python /mnt/SSD3/xinyi/benchmark/task1_internvl3_5_8B.py \
  --gpu 1 \
  --tp 1 \
  --max_frames 60 \
  --max_videos -1 \
  --disable_logs True \
  --max_retries 10 \
  --max_new_tokens 2048 \
  --model_name OpenGVLab/InternVL3_5-8B \
  --dataset_dir /mnt/SSD3/tengyou/seizure_videos/segments/all_dataset/ \
  --output_dir  /mnt/SSD3/xinyi/benchmark/ \
  --cache_dir   /mnt/SSD3/xinyi/benchmark/model_cache

  

