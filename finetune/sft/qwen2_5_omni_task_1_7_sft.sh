#!/bin/bash

VERSION_PREFIX="qwen_2.5_omni_task17_sft"
TIMESTAMP="$(date +%Y%m%d%H%M%S)"
VERSION_NAME="${VERSION_PREFIX}_${TIMESTAMP}"
LOG_PATH="./run_logs/${VERSION_NAME}.log"

# todo
#BASE_MODEL='../ckpts/init_models/Qwen2.5-Omni-3B'
#BASE_MODEL= 'Qwen/Qwen2.5-Omni-7B'
# todo: data_path
export TASK_DATASET='./dataset/sft_merge_2025-11-01_swift_train.jsonl'

export WANDB_API_KEY="***REMOVED***"   # 或提前 wandb login
export WANDB_PROJECT="seizurebench"
export WANDB_ENTITY="linazhang-ucla"

# todo: model_path
MODEL_SAVE_PATH='./ckpts/trained_models/'${VERSION_NAME}

export SWIFT_CHANNEL_POLICY='{
  "task-4":  {"mode":"fps", "fps":1},
  "task-7-1":{"mode":"uniform", "num_frames":60},
  "task-7-2":{"mode":"uniform", "num_frames":60},
  "default": {"mode":"fps", "fps":2}
}'

mkdir -p './run_logs/'
mkdir -p './ckpts/'
mkdir -p './ckpts/trained_models/'

#VIDEO_MAX_PIXELS=$((784*448)) \
#VIDEO_MAX_PIXELS=$((392*448)) \
# --dataloader_num_workers 4 \

export SAMPLING_RATE=16000

PYTHONWARNINGS='ignore:PySoundFile failed.*:UserWarning:swift.llm.template.template.qwen,ignore:__audioread_load.*:FutureWarning:librosa.core.audio' \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
VIDEO_MAX_PIXELS=$((784*448)) \
FPS_MAX_FRAMES=60 \
FPS=1 \
MAX_PIXELS=$((FPS_MAX_FRAMES * VIDEO_MAX_PIXELS)) \
swift sft \
    --model "Qwen/Qwen2.5-Omni-7B"\
    --model_kwargs '{"use_audio_in_video": true}' \
    --dataset $TASK_DATASET  \
    --load_from_cache_file true \
    --split_dataset_ratio 0.05 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --attn_impl sdpa \
    --gradient_checkpointing true \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_accumulation_steps 4 \
    --eval_strategy steps --eval_steps 100 \
    --save_strategy steps --save_steps 100 --save_total_limit 3 \
    --logging_steps 20 \
    --max_length 32768 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 1 \
    --load_best_model_at_end true \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --deepspeed zero2 \
    --report_to 'wandb' \
    --run_name "$VERSION_NAME" \
    --enable_channel_loss 'True' \
    --output_dir $MODEL_SAVE_PATH  2>&1 | tee "$LOG_PATH"

    