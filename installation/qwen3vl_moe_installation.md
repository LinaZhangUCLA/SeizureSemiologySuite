# Environment Installation

```shell
# If you already have the qwenvl (for qwen2.5vl) conda environment, you can clone it.
# Reinstall the pinned PyTorch stack below before building FlashAttention, otherwise
# the cloned env may keep an incompatible CUDA wheel/toolkit combination.
conda create --name qwen3vl_moe  --clone qwenvl
conda activate qwen3vl_moe
python -m pip install transformers==4.57.0

# If you run task34567 you need this package. Task12 does not.
python -m pip install qwen-vl-utils==0.0.14
```


```shell
# Install from the beginning
conda create -n qwen3vl_moe python=3.10 -y
conda activate qwen3vl_moe
python -m pip install -U pip

# Match the PyTorch wheel line to your local CUDA toolkit (`nvcc --version`).
# Example below is for CUDA 12.8. If your toolkit is different, replace the
# wheel line with the matching command from:
# https://pytorch.org/get-started/previous-versions/
python -m pip install \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 torchcodec==0.11.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
  transformers==4.57.0 accelerate qwen-vl-utils==0.0.14 pandas peft tqdm numpy scipy datasets deepspeed

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

# Requires ffmpeg if there is no ffmpeg on your server

```shell
sudo cp /var/lib/dpkg/statoverride /var/lib/dpkg/statoverride.bak
sudo sed -i '/messagebus/d' /var/lib/dpkg/statoverride

sudo locale-gen en_US.UTF-8
sudo update-locale LANG=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

sudo apt update
sudo apt --fix-broken install -y

sudo apt install -y ffmpeg

ffmpeg -version
``` 
