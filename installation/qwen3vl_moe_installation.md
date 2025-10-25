# Environment Installation

```shell
# If you already have the qwenvl(for qwen2.5vl) conda environment, follow these steps to save time.
conda create --name qwen3vl_moe  --clone qwenvl
conda activate qwen3vl_moe
pip install transformers==4.57
```


```shell
# install from the begining
conda create -n qwen3vl_moe python=3.10 -y
conda activate qwen3vl_moe

# CUDA 12.4
pip install torch torchvision torchaudio torchcodec \
  transformers==4.57.0 accelerate qwen-vl-utils pandas peft tqdm numpy scipy datasets deepspeed \
  --extra-index-url https://download.pytorch.org/whl/cu124

# CUDA 12.2 (use cu121 wheel, cu122 may fail)
pip install torch torchvision torchaudio torchcodec \
  transformers==4.57.0 accelerate qwen-vl-utils pandas peft tqdm numpy scipy datasets deepspeed \
  --extra-index-url https://download.pytorch.org/whl/cu121


# FlashAttention Installation (ABI Compatibility Fix)
pip install --upgrade setuptools wheel && \
pip uninstall -y flash-attn || true && \
pip cache purge && \
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn --no-build-isolation --no-cache-dir
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
