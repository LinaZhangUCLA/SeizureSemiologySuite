# Environment Installation(for Audio_Flamingo_3)
```shell
conda create -n af3_env python=3.10 -y
conda activate af3_env
python -m pip install -U pip setuptools wheel
pip install uv
alias uvp="uv pip"

# CUDA 12.2 (use cu121 wheel, cu122 may fail) 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.8 (choose one CUDA version based on your setup)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128


# Clone Audio-Flamingo_3 repo and llava
git clone https://github.com/NVIDIA/audio-flamingo.git
cd audio-flamingo
git checkout audio_flamingo_3

# This is required to enable PEP 660 support
pip install --upgrade pip setuptools
pip install -e ".[train,eval]"

# Install FlashAttention2 and related packages
pip install \
  hydra-core loguru Pillow pydub \
  pandas tqdm scipy datasets \
  flash_attn==2.7.3 \
  transformers==4.46.0 \
  accelerate==0.34.2 \
  deepspeed==0.15.4 \
  numpy==1.26.4 \
  opencv-python-headless==4.8.0.76 \
  matplotlib \
  peft==0.14.0

# audio
pip install soundfile librosa openai-whisper ftfy jiwer kaldiio wandb
conda install -c conda-forge ffmpeg -y

site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')

# Downgrade protobuf to 3.20 for backward compatibility
pip install protobuf==3.20.*

# Replace transformers and deepspeed files
cp -rv ./llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/
```