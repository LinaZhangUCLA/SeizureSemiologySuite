# Environment setup
```shell
conda create -n omni python=3.10 -y
conda activate omni

conda install -c conda-forge ffmpeg -y
python -m pip install -U pip

# Match the PyTorch wheel line to your local CUDA toolkit (`nvcc --version`).
# Example below is for CUDA 12.8. If your toolkit is different, replace the
# wheel line with the matching command from:
# https://pytorch.org/get-started/previous-versions/
python -m pip install \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install -U decord==0.6.0 opencv-python ffmpeg-python pydub "librosa>=0.10,<0.11" pandas tqdm requests numpy "accelerate>=0.34,<1" qwen-omni-utils "transformers>=4.50,<5" "huggingface-hub>=0.23" pillow

# Used only for tasks3456. Unnecessary for task12.
python -m pip install -U qwen-vl-utils peft
```

# Quick import check (not necessary)
```shell
python - <<'PY'
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
print("Qwen2.5-Omni imports OK")
PY
```
