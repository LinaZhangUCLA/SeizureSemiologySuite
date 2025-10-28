#!/bin/bash
# ================================================================
# Created by: Jungang
# Email: ljungang.02@gmail.com
# Description: fine-tuning command with qwen2.5_omni_7b.
# ================================================================
# todo: data_path
VERSION_NAME='qwen_2.5_omni_task_1_7_sft_20251024'
VERSION_NAME='qwen_2.5_omni_task_1_7_sft_20251028'
LOG_PATH="./run_logs/${VERSION_NAME}.log"

# todo
BASE_MODEL='./ckpts/init_models/Qwen2.5-Omni-7B'
#BASE_MODEL= 'Qwen/Qwen2.5-Omni-7B'

# todo: data_path
export TASK_DATASET='./dataset/test.jsonl'

# todo: model_path
MODEL_SAVE_PATH='./ckpts/trained_models/'${VERSION_NAME}

export SWIFT_CHANNEL_POLICY='{
  "task-4":  {"mode":"fps", "fps":1},
  "task-7-1":{"mode":"uniform", "num_frames":32},
  "task-7-2":{"mode":"uniform", "num_frames":32},
  "default": {"mode":"fps", "fps":2}
}'

mkdir -p ./run_logs/

# --model_type "qwen2_5" \
CUDA_VISIBLE_DEVICES=3 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
swift sft \
  --train_type full \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 2e-5 \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --freeze_llm false \
  --gradient_accumulation_steps 8 \
  --eval_steps -1 \
  --save_steps 1 \
  --save_total_limit 5 \
  --logging_steps 1 \
  --max_length 16384 \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --model $BASE_MODEL \
  --dataset ssb \
  --split_dataset_ratio 0 \
  --init_weights 'True' \
  --attn_impl 'sdpa' \
  --report_to 'swanlab' \
  --enable_channel_loss 'True' \
  --swanlab_project 'swift-robot' \
  --custom_register_path ./fintune/utils/custom_sample/dataset.py \
  --output_dir $MODEL_SAVE_PATH > $LOG_PATH 2>&1 &
