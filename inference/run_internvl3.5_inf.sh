set -Eeuo pipefail

### ---------- User-configurable variables ----------
CONDA_ENV="internvl3_5"  # your conda environment name
PY_SCRIPT="/mnt/SSD3/xinyi/benchmark/task1_internvl3.5_8B.py"

GPU="2"
TP="1"
MAX_FRAMES="60"
MAX_VIDEOS="-1"
DISABLE_LOGS="True"
MAX_RETRIES="10"
MAX_NEW_TOKENS="2048"

MODEL_NAME="OpenGVLab/InternVL3_5-8B"
DATASET_DIR="/mnt/SSD3/tengyou/seizure_videos/segments/all_dataset/"
OUTPUT_DIR="/mnt/SSD3/xinyi/benchmark/"
CACHE_DIR="/mnt/SSD3/xinyi/benchmark/model_cache"

# Optional: redirect HF cache if you want (uncomment to use)
# export HF_HOME="${CACHE_DIR}/hf_home"

### ---------- Helpers ----------
log() { printf "[%(%F %T)T] %s\n" -1 "$*"; }

### ---------- Activate conda env ----------
if command -v conda >/dev/null 2>&1; then
  # Ensure 'conda activate' works in non-interactive shells
  eval "$(conda shell.bash hook)"
  log "Activating conda env: ${CONDA_ENV}"
  conda activate "${CONDA_ENV}"
else
  echo "ERROR: 'conda' command not found. Install Miniconda/Anaconda or adjust the script." >&2
  exit 1
fi

log "Python: $(python --version) at $(which python)"

### ---------- Ensure huggingface CLI is available ----------
if ! command -v huggingface-cli >/dev/null 2>&1; then
  log "Installing huggingface_hub to get huggingface-cli ..."
  pip install -U huggingface_hub >/dev/null
fi

### ---------- Login to Hugging Face ----------
# Non-interactive: if HUGGING_FACE_HUB_TOKEN is set, use it directly.
if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  log "Logging into Hugging Face using HUGGING_FACE_HUB_TOKEN (non-interactive)."
  huggingface-cli login --token "${HUGGING_FACE_HUB_TOKEN}" --add-to-git-credential >/dev/null
else
  # Interactive fallback: prompt securely once.
  log "HUGGING_FACE_HUB_TOKEN not set; prompting for token (input hidden)."
  read -rs -p "Enter your Hugging Face token (starts with hf_): " HF_TOKEN
  echo
  if [[ -z "${HF_TOKEN}" ]]; then
    echo "ERROR: Empty token provided." >&2
    exit 1
  fi
  huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential >/dev/null
fi

# Show current identity (non-fatal if it fails)
huggingface-cli whoami || true

### ---------- Run the Python task ----------
log "Launching InternVL3.5 job..."
python "${PY_SCRIPT}" \
  --gpu "${GPU}" \
  --tp "${TP}" \
  --max_frames "${MAX_FRAMES}" \
  --max_videos "${MAX_VIDEOS}" \
  --disable_logs "${DISABLE_LOGS}" \
  --max_retries "${MAX_RETRIES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --model_name "${MODEL_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --cache_dir "${CACHE_DIR}"

log "Job finished."
  


