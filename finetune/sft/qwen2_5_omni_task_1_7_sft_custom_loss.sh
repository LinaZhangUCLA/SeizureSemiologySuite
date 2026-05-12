#!/bin/bash

VERSION_PREFIX="qwen_2.5_omni_task17_sft_custom_loss"
TIMESTAMP="$(date +%Y%m%d%H%M%S)"
VERSION_NAME="${VERSION_PREFIX}_${TIMESTAMP}"
LOG_PATH="./run_logs/${VERSION_NAME}.log"

export TASK_DATASET='./dataset/sft_merge_2026-03-26_swift_train.jsonl'

# 模型缓存目录（避免下载到 /home 的默认路径）
#export MODELSCOPE_CACHE='/mnt/SSD3/lina/SeizureSemiologyBench/cache/modelscope'
#export HF_HOME='/mnt/SSD3/lina/SeizureSemiologyBench/cache/huggingface'

export WANDB_API_KEY="$WANDB_API_KEY"
export WANDB_PROJECT="seizurebench"
export WANDB_ENTITY="linazhang-ucla"

MODEL_SAVE_PATH='./ckpts/trained_models/'${VERSION_NAME}

# export SWIFT_CHANNEL_POLICY='{
#   "task-4":  {"mode":"fps", "fps":1},
#   "task-7-1":{"mode":"uniform", "num_frames":60},
#   "task-7-2":{"mode":"uniform", "num_frames":60},
#   "default": {"mode":"fps", "fps":2}
# }'

# ── Task-4 custom loss hyperparameters ────────────────────────────────────────
# Idea 2: CE scale for task-4 samples (short responses → less gradient by default)
export TASK4_CE_SCALE=2.0
# Idea 1: extra weight on the 4 MM/SS digit positions in task-4 CE loss
export TASK4_DIGIT_W=2.0
# Idea 3: weight of the soft-MAE auxiliary term (units: same as CE loss)
export TASK4_MAE_LAMBDA=0.05
# Idea 3: smooth-L1 transition threshold in seconds (errors < beta treated as L2)
export TASK4_MAE_BETA=10.0
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p './run_logs/'
mkdir -p './ckpts/'
mkdir -p './ckpts/trained_models/'

export SAMPLING_RATE=16000

#VIDEO_MAX_PIXELS=$((784*448)) \
#VIDEO_MAX_PIXELS=$((392*448)) \

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"


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
    --use_hf true \
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
    --external_plugins "${SCRIPT_DIR}/plugin_sft.py" \
    --loss_type task4_combined \
    --output_dir $MODEL_SAVE_PATH  2>&1 | tee "$LOG_PATH"
