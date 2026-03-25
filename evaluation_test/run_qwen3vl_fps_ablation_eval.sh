#!/usr/bin/env bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-/mnt/SSD3/lina/my_conda_env/qwen3vl_moe}"
MODEL_TAG="${MODEL_TAG:-Qwen3-VL-32B-Instruct}"
VIDEO_RANGE="${VIDEO_RANGE:-1-300}"
FPS_LIST_STRING="${FPS_LIST:-1 4}"
INPUT_ROOT="${INPUT_ROOT:-${REPO_ROOT}/result/fps_ablation/qwen3_vl_32b}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/metrics_test/fps_ablation/qwen3_vl_32b}"
CACHE_DIR="${CACHE_DIR:-/mnt/SSD3/lina/SeizureSemiologyBench/cache}"
CACHE_FALLBACK_DIR="${CACHE_FALLBACK_DIR:-${REPO_ROOT}/.cache/qwen3vl}"
LOCAL_BERT_MODEL="${LOCAL_BERT_MODEL:-/mnt/SSD3/lina/bertmodel}"
TASK2_METRICS_DEVICE="${TASK2_METRICS_DEVICE:-cuda:2}"
TASK6_METRICS_DEVICE="${TASK6_METRICS_DEVICE:-cuda:3}"
RUN_TASK2_QLIST="${RUN_TASK2_QLIST:-0}"
RUN_TASK6_RQI="${RUN_TASK6_RQI:-0}"

GT_TASK12="${REPO_ROOT}/result/ground_truth/task12_annotation.csv"
GT_TASK5="${REPO_ROOT}/result/ground_truth/task5_sequence_annotation_vlm.csv"
GT_TASK6="${REPO_ROOT}/result/ground_truth/task6_report_annotation.csv"

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

if [[ "${RUN_TASK2_QLIST}" == "1" ]]; then
    if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
        echo "RUN_TASK2_QLIST=1 requires DASHSCOPE_API_KEY." >&2
        exit 1
    fi

    GT_Q_DIR="${OUTPUT_ROOT}/ground_truth/task2_question_list"
    GT_Q_RAW="${GT_Q_DIR}/task12_ground_truth_question_list.csv"
    GT_Q_MERGED="${GT_Q_DIR}/task12_ground_truth_question_list_with_yesno.csv"
    mkdir -p "${GT_Q_DIR}"

    if [[ ! -f "${GT_Q_RAW}" ]]; then
        python "${SCRIPT_DIR}/task2_question-list-csv.py" \
            --input-csv "${GT_TASK12}" \
            --output-csv "${GT_Q_RAW}" \
            --api-key "${DASHSCOPE_API_KEY}"
    fi

    if [[ ! -f "${GT_Q_MERGED}" ]]; then
        python "${SCRIPT_DIR}/task2_merge_yesno_into_qlist-csv.py" \
            --inputs "${GT_TASK12}" \
            --qlist "${GT_Q_RAW}" \
            --out-dir "${GT_Q_DIR}" \
            --out-name "$(basename "${GT_Q_MERGED}")"
    fi
fi

read -r -a FPS_VALUES <<< "${FPS_LIST_STRING}"

for fps in "${FPS_VALUES[@]}"; do
    fps_tag="${fps//./p}"
    task12_dir="${INPUT_ROOT}/task12_fps_${fps_tag}"
    task56_dir="${INPUT_ROOT}/task56_fps_${fps_tag}"
    fps_out_dir="${OUTPUT_ROOT}/fps_${fps_tag}"
    mkdir -p "${fps_out_dir}"

    task12_raw="${task12_dir}/Task1_${MODEL_TAG}_${VIDEO_RANGE}.csv"
    task12_merged="${task12_dir}/Task12_${MODEL_TAG}_${VIDEO_RANGE}_merged.csv"
    task12_llmmerge="${task12_dir}/Task12_${MODEL_TAG}_${VIDEO_RANGE}_merged_llmmerge.csv"
    task5_raw="${task56_dir}/Task5_${MODEL_TAG}_${VIDEO_RANGE}.csv"
    task6_raw="${task56_dir}/Task6_${MODEL_TAG}_${VIDEO_RANGE}.csv"
    task6_merged="${task56_dir}/Task6_${MODEL_TAG}_${VIDEO_RANGE}_merged.csv"

    echo "=== Task 1 | fps=${fps} ==="
    python "${SCRIPT_DIR}/Task1_segment_merge.py" \
        --input_csv "${task12_raw}" \
        --output_csv "${task12_merged}"

    python "${SCRIPT_DIR}/task1_compute_metrics.py" \
        --pred_csv "${task12_merged}" \
        --gt_csv "${GT_TASK12}" \
        --model_name "${MODEL_TAG}_fps_${fps_tag}" \
        --out_csv "${fps_out_dir}/task1_metrics.csv"

    echo "=== Task 2 | fps=${fps} ==="
    python "${SCRIPT_DIR}/Task2_llmmerge_qwen.py" \
        --input_csv "${task12_merged}" \
        --output_csv "${task12_llmmerge}"

    python "${SCRIPT_DIR}/Task2_blue_rouge_bertscore_metrics.py" \
        --pred_csv "${task12_llmmerge}" \
        --gt_csv "${GT_TASK12}" \
        --out_dir "${fps_out_dir}/task2_feature_metrics" \
        --local_model_path "${LOCAL_BERT_MODEL}" \
        --device "${TASK2_METRICS_DEVICE}"

    if [[ "${RUN_TASK2_QLIST}" == "1" ]]; then
        qlist_dir="${fps_out_dir}/task2_question_list"
        pred_q_raw="${qlist_dir}/${MODEL_TAG}_fps_${fps_tag}_question_list.csv"
        pred_q_merged="${qlist_dir}/${MODEL_TAG}_fps_${fps_tag}_question_list_with_yesno.csv"
        mkdir -p "${qlist_dir}"

        python "${SCRIPT_DIR}/task2_question-list-csv.py" \
            --input-csv "${task12_llmmerge}" \
            --output-csv "${pred_q_raw}" \
            --api-key "${DASHSCOPE_API_KEY}"

        python "${SCRIPT_DIR}/task2_merge_yesno_into_qlist-csv.py" \
            --inputs "${task12_merged}" \
            --qlist "${pred_q_raw}" \
            --out-dir "${qlist_dir}" \
            --out-name "$(basename "${pred_q_merged}")"

        python "${SCRIPT_DIR}/task2_question_list_matrics.py" \
            --gt-csv "${GT_Q_MERGED}" \
            --vlm-csv "${pred_q_merged}" \
            --model-name "${MODEL_TAG}_fps_${fps_tag}" \
            --out-dir "${qlist_dir}/metrics"
    fi

    echo "=== Task 5 | fps=${fps} ==="
    python "${SCRIPT_DIR}/task5_merge_segment.py" \
        --input_csv "${task5_raw}"

    python "${SCRIPT_DIR}/task5_compute_sequence_metrics.py" \
        --pred_csv "${task56_dir}/Task5_${MODEL_TAG}_${VIDEO_RANGE}_merge.csv" \
        --gt_csv "${GT_TASK5}" \
        --out_csv "${fps_out_dir}/task5_per_video_metrics.csv" \
        --model_name "${MODEL_TAG}_fps_${fps_tag}" \
        --summary_csv "${fps_out_dir}/task5_summary_metrics.csv"

    echo "=== Task 6 | fps=${fps} ==="
    python "${SCRIPT_DIR}/Task6_llmmerge_segment.py" \
        --input_csv "${task6_raw}" \
        --output_csv "${task6_merged}"

    python "${SCRIPT_DIR}/Task6_model_metrics.py" \
        --pred_csv "${task6_merged}" \
        --gt_csv "${GT_TASK6}" \
        --model_name "${MODEL_TAG}_fps_${fps_tag}" \
        --out_csv "${fps_out_dir}/task6_nlp_metrics.csv" \
        --device "${TASK6_METRICS_DEVICE}"

    if [[ "${RUN_TASK6_RQI}" == "1" ]]; then
        if [[ -z "${OPENAI_API_KEY:-}" ]]; then
            echo "RUN_TASK6_RQI=1 requires OPENAI_API_KEY." >&2
            exit 1
        fi

        python "${SCRIPT_DIR}/seizure_rqi_evaluation.py" \
            --vlm_csv "${task6_merged}" \
            --gt_csv "${GT_TASK6}" \
            --output "${fps_out_dir}/task6_rqi.csv" \
            --model "gpt-4o"
    fi
done

echo "Ablation evaluation completed. Outputs are under ${OUTPUT_ROOT}."
