#!/bin/bash

VERSION_PREFIX="qwen_2.5_omni_task_1_7_sft"
TIMESTAMP="$(date +%Y%m%d%H%M%S)"
VERSION_NAME="${VERSION_PREFIX}_${TIMESTAMP}"


# todo: data_path
# VERSION_NAME='qwen_2.5_omni_task_1_7_sft_20251024'
# VERSION_NAME='qwen_2.5_omni_task_1_7_sft_20251025'

LOG_PATH="./run_logs/${VERSION_NAME}.log"

# todo
BASE_MODEL='./ckpts/init_models/Qwen2.5-Omni-3B'
#BASE_MODEL= 'Qwen/Qwen2.5-Omni-7B'

# todo: data_path
#export TASK_DATASET='./dataset/test.jsonl'
export TASK_DATASET='./dataset/sft_merge_2025-10-26_swift.jsonl'

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

# CUDA_VISIBLE_DEVICES=3 \
# VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
# CUDA_VISIBLE_DEVICES=1,2,3 NPROC_PER_NODE=3 MASTER_PORT=29513 \
# # VIDEO_MAX_PIXELS=50176 \
# # FPS_MAX_FRAMES=12 \
# # MAX_PIXELS=1003520 \
# swift sft \
#   --model "$BASE_MODEL" \
#   --use_hf true \
#   #--dataset "$TASK_DATASET" \
#   --dataset ssb \
#   --split_dataset_ratio 0.01 \
#   --tuner_backend peft \
#   --train_type lora \
#   --torch_dtype bfloat16 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --gradient_checkpointing true \
#   --lr_scheduler_type cosine \
#   --learning_rate 5e-5 \
#   --lora_rank 8 --lora_alpha 32 --lora_dropout 0.2 --target_modules all-linear \
#   --num_train_epochs 3 \
#   --gradient_accumulation_steps 16 \
#   --save_strategy steps --save_steps 200 \
#   --eval_strategy steps --eval_steps 200 \
#   --save_total_limit 5 \
#   --padding_free false \
#   --attn_impl sdpa \
#   --logging_steps 1 --log_level info --logging_first_step true \
#   --max_length 32768 \
#   --warmup_ratio 0.01 \
#   --dataset_num_proc 4 \
#   --dataloader_num_workers 4 \
#   --dataloader_pin_memory false \
#   --dataloader_persistent_workers false \
#   --dataloader_drop_last false \
#   --logging_nan_inf_filter false \
#   --dataset_shuffle true --train_dataloader_shuffle true \
#   --skip_memory_metrics true \
#   --remove_unused_columns false \
#   --metric_for_best_model eval_loss --greater_is_better false --load_best_model_at_end true \
#   --report_to wandb \
#   --enable_channel_loss true \
#   --deepspeed zero2 \
 
#   --custom_register_path ./finetune/utils/custom_sample/dataset.py \
#   --output_dir "$MODEL_SAVE_PATH" 2>&1 | tee "$LOG_PATH"

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
  --report_to 'wandb' \
  --enable_channel_loss 'True' \
  --custom_register_path ./finetune/utils/custom_sample/dataset.py \
  --output_dir $MODEL_SAVE_PATH > $LOG_PATH #2>&1 &

