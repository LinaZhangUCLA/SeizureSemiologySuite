#!/bin/bash

VERSION_PREFIX="qwen_2.5_omni_task1_grpo"
TIMESTAMP="$(date +%Y%m%d%H%M%S)"
VERSION_NAME="${VERSION_PREFIX}_${TIMESTAMP}"
LOG_PATH="./run_logs/${VERSION_NAME}.log"

# todo model_path
BASE_MODEL='/mnt/SSD1/linazhang/seizure_finetune/seizure_omni_sft'
MODEL_SAVE_PATH='./ckpts/trained_models/'${VERSION_NAME}

# todo: data_path
export TASK_DATASET='./dataset/grpo_task1only_2026-03-17_swift_train.jsonl'



mkdir -p './run_logs/'
mkdir -p './ckpts/'
mkdir -p './ckpts/trained_models/'

#VIDEO_MAX_PIXELS=$((784*448)) \
#VIDEO_MAX_PIXELS=$((392*448)) \
# --dataloader_num_workers 4 \

export WANDB_API_KEY="***REMOVED***"   # 或提前 wandb login
export WANDB_PROJECT="seizurebench_grpo"
export WANDB_ENTITY="linazhang-ucla"

export ENABLE_AUDIO_OUTPUT=False

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
VIDEO_MAX_PIXELS=$((288*288)) \
FPS_MAX_FRAMES=60 \
MAX_PIXELS=$((FPS_MAX_FRAMES * VIDEO_MAX_PIXELS)) \
NPROC_PER_NODE=2 \
ENABLE_AUDIO_OUTPUT=1 \
CUDA_VISIBLE_DEVICES=2,3 \
FPS=1 \
swift rlhf \
    --rlhf_type grpo \
    --model $BASE_MODEL \
    --model_kwargs '{"use_audio_in_video": true}' \
    --reward_funcs seizure \
    --reward_weights 1.0 \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset $TASK_DATASET  \
    --load_from_cache_file true \
    --external_plugins './rlhf/plugin_task1.py' \
    --max_steps 6000 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 1 \
    --eval_steps 300 \
    --save_steps 100 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --max_new_tokens 1024 \
    --max_length 32768 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --num_generations 4 \
    --temperature 0.8 \
    --top_p 0.95 \
    --top_k 50 \
    --system './rlhf/prompt.txt' \
    --deepspeed zero2 \
    --log_completions true \
    --report_to 'wandb' \
    --run_name "$VERSION_NAME" \
    --enable_channel_loss 'True' \
    --output_dir $MODEL_SAVE_PATH  2>&1 | tee "$LOG_PATH"
