# Environment setup
conda create -n omni python=3.10 -y
conda activate omni


conda install -c conda-forge ffmpeg -y
pip install ffmpeg-python

pip install --upgrade pip

pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

pip install -U "transformers==4.43.3" "huggingface-hub>=0.23.0" safetensors Pillow


pip install decord==0.6.0 opencv-python ffmpeg-python pydub "librosa>=0.10,<0.11"


pip install -U pandas tqdm requests numpy

pip install -U "accelerate>=0.34,<1"
pip install -U qwen-omni-utils


pip uninstall -y transformers
pip install -U "git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview"
pip install -U qwen-vl-utils peft

# Quick import check (not necessary)
python - <<'PY'
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
print("Qwen2.5-Omni imports OK")
PY

