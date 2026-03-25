#!/usr/bin/env bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-/mnt/SSD3/lina/my_conda_env/qwen3vl_moe}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-32B-Instruct}"
CACHE_DIR="${CACHE_DIR:-/mnt/SSD3/lina/SeizureSemiologyBench/cache}"
CACHE_FALLBACK_DIR="${CACHE_FALLBACK_DIR:-${REPO_ROOT}/.cache/qwen3vl}"
TASK12_DATASET="${TASK12_DATASET:-/mnt/SSD3/lina/ucla2/ssbench/task1256_segment_60s}"
TASK56_DATASET_ROOT="${TASK56_DATASET_ROOT:-/mnt/SSD3/lina/ucla2/ssbench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/result/fps_ablation/qwen3_vl_32b}"
VIDEO_RANGE="${VIDEO_RANGE:-1-300}"
FPS_LIST_STRING="${FPS_LIST:-1 4}"
GPU12="${GPU12:-0,1}"
GPU56="${GPU56:-0,1}"

# Keep the original 120-frame budget by default.
# If you want full 60 s coverage for Task 1/2 at 4 FPS, set TASK12_MAX_FRAMES_FPS4=240.
TASK12_MAX_FRAMES="${TASK12_MAX_FRAMES:-120}"
TASK12_MAX_FRAMES_FPS4="${TASK12_MAX_FRAMES_FPS4:-${TASK12_MAX_FRAMES}}"
TASK56_MAX_FRAMES="${TASK56_MAX_FRAMES:-120}"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "Could not find conda initialization script." >&2
    exit 1
fi

conda activate "${CONDA_ENV}"
set -u

if ! mkdir -p "${CACHE_DIR}" 2>/dev/null || [[ ! -w "${CACHE_DIR}" ]]; then
    echo "Cache dir is not writable: ${CACHE_DIR}" >&2
    echo "Falling back to writable cache dir: ${CACHE_FALLBACK_DIR}" >&2
    CACHE_DIR="${CACHE_FALLBACK_DIR}"
fi

mkdir -p "${CACHE_DIR}"
mkdir -p "${OUTPUT_ROOT}"
echo "Using cache dir: ${CACHE_DIR}"

read -r -a FPS_VALUES <<< "${FPS_LIST_STRING}"

for fps in "${FPS_VALUES[@]}"; do
    fps_tag="${fps//./p}"
    task12_out="${OUTPUT_ROOT}/task12_fps_${fps_tag}"
    task56_out="${OUTPUT_ROOT}/task56_fps_${fps_tag}"
    task12_frames="${TASK12_MAX_FRAMES}"

    if [[ "${fps}" == "4" ]]; then
        task12_frames="${TASK12_MAX_FRAMES_FPS4}"
    fi

    mkdir -p "${task12_out}" "${task56_out}"

    echo "=== Task 1/2 | fps=${fps} | max_frames=${task12_frames} ==="
    python "${SCRIPT_DIR}/task12_Qwen3-VL-32B-Instruct.py" \
        --gpu "${GPU12}" \
        --model_name "${MODEL_NAME}" \
        --dataset_dir "${TASK12_DATASET}" \
        --cache_dir "${CACHE_DIR}" \
        --output_dir "${task12_out}" \
        --videos_range "${VIDEO_RANGE}" \
        --fps "${fps}" \
        --max_frames "${task12_frames}"

    echo "=== Task 5/6 | fps=${fps} | max_frames=${TASK56_MAX_FRAMES} ==="
    python "${SCRIPT_DIR}/task34567_Qwen3VL_new_jiarui.py" \
        --gpu "${GPU56}" \
        --model_name "${MODEL_NAME}" \
        --dataset_dir "${TASK56_DATASET_ROOT}" \
        --cache_dir "${CACHE_DIR}" \
        --output_dir "${task56_out}" \
        --videos_range "${VIDEO_RANGE}" \
        --run_tasks 5,6 \
        --fps "${fps}" \
        --max_frames "${TASK56_MAX_FRAMES}"
done

echo "Ablation runs completed. Outputs are under ${OUTPUT_ROOT}."
