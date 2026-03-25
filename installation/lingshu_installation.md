## Environment Installation (for Lingshu + Qwen2.5-VL)

```bash
# If you already have the qwenvl conda environment, follow these steps to save time.
conda create --name lingshu  --clone qwenvl
conda activate lingshu 
conda install -c conda-forge ffmpeg -y 
pip install ffmpeg-python
```

```bash
# 1. Create new conda env
conda create -n lingshu python=3.10 -y
conda activate lingshu
python -m pip install -U pip

# 2. Install ffmpeg (for video reading)
conda install -c conda-forge ffmpeg -y
python -m pip install ffmpeg-python

# 3. Install PyTorch from the wheel line that matches `nvcc --version`.
# Example below is for CUDA 12.8. If your toolkit is different, replace the
# wheel line with the matching command from:
# https://pytorch.org/get-started/previous-versions/
python -m pip install \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
  transformers==4.51.3 accelerate qwen-vl-utils pandas peft tqdm numpy scipy
python -m pip install -U peft

# 4. Install FlashAttention
python -m pip install -U "packaging" "psutil" "ninja" "wheel" "setuptools<82"
python -m pip uninstall -y flash-attn || true
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
PY
nvcc --version
MAX_JOBS=8 FLASH_ATTENTION_FORCE_BUILD=TRUE python -m pip install flash-attn --no-build-isolation --no-cache-dir
```
