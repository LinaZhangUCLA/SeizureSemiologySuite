# Environment Installation (for Qwen2.5-VL script)

```shell
conda create -n qwenvl python=3.10 -y
conda activate qwenvl
python -m pip install -U pip

# Match the PyTorch wheel line to your local CUDA toolkit (`nvcc --version`).
# Example below is for CUDA 12.8. If your toolkit is different, replace the
# wheel line with the matching command from:
# https://pytorch.org/get-started/previous-versions/
python -m pip install \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 torchcodec==0.11.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
  transformers==4.51.3 accelerate qwen-vl-utils pandas peft tqdm numpy scipy datasets deepspeed

# FlashAttention must see the same CUDA minor version from both PyTorch and nvcc.
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
