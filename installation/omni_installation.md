# 1. Create and activate a fresh conda env (Python 3.10)
conda create -n omni python=3.10 -y
conda activate omni

# 2. Install ffmpeg (for audio/video I/O)
conda install -c conda-forge ffmpeg -y
pip install ffmpeg-python

# 3. Install PyTorch (pick wheel matching your CUDA; check with `nvidia-smi`)
pip install --upgrade pip
# CUDA 12.x:
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
# If your server is CUDA 11.x, use cu118 instead:
# pip install --extra-index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio

# 4. (Initial) HF core libs you listed  — NOTE: no torchvision here to avoid overriding the CUDA wheel
pip install -U "transformers==4.43.3" "huggingface-hub>=0.23.0" safetensors Pillow

# 5. Video / image / audio tooling
pip install decord==0.6.0 opencv-python ffmpeg-python pydub "librosa>=0.10,<0.11"

# 6. Extra deps needed by the big script
pip install -U pandas tqdm requests numpy

# 7. Omni-specific runtime
pip install -U "accelerate>=0.34,<1"
pip install -U qwen-omni-utils

# 8. Switch transformers to Qwen2.5-Omni preview branch (per model card)
pip uninstall -y transformers
pip install -U "git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview"

# 9. Quick import check
python - <<'PY'
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
print("Qwen2.5-Omni imports OK")
PY

