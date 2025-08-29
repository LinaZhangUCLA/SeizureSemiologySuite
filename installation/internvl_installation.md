conda create -n internvl3_5 python=3.10 -y
conda activate internvl3_5
python -m pip install -U pip

#cuda11.8
pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
  torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118

#cuda 12.2
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.7.1+cu121 torchvision==0.22.1+cu121 torchaudio==2.7.1+cu121

pip install lmdeploy==0.9.2.post1 transformers==4.51.0 huggingface-hub==0.33.2 \
  accelerate==1.8.1 safetensors==0.5.3 tokenizers==0.21.2 timm==1.0.16 einops==0.8.1 \
  decord==0.6.0 pillow==11.0.0 numpy==1.26.4 pandas==2.3.1 tqdm==4.67.1 requests==2.32.4 \
  PyYAML==6.0.2

