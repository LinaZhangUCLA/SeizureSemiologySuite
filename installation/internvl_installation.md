# Environment Installation (for Internvl3_5 script)

```shell
conda create -n internvl3_5 python=3.10 -y
conda activate internvl3_5
python -m pip install -U pip

# Match the PyTorch wheel line to your local CUDA toolkit (`nvcc --version`).
# Example below is for CUDA 12.8. If your toolkit is different, replace the
# wheel line with the matching command from:
# https://pytorch.org/get-started/previous-versions/
python -m pip install \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 torchcodec==0.11.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install lmdeploy==0.9.2.post1 transformers==4.51.0 huggingface-hub==0.33.2 \
  accelerate==1.8.1 safetensors==0.5.3 tokenizers==0.21.2 timm==1.0.16 einops==0.8.1 \
  decord==0.6.0 pillow==11.0.0 numpy==1.26.4 pandas==2.3.1 tqdm==4.67.1 requests==2.32.4 \
  PyYAML==6.0.2
```

# Login huggingface
```shell
huggingface-cli login # When prompted, paste your token (e.g., hf_xxx...) and press Enter to confirm.
```

# To get huggingface token: 
```shell
#Sign in at 
huggingface.co → click your avatar → Settings → Access Tokens → New token. 
#Token type: 
choose Fine-grained (cannot be changed after creation). 
#Token name:
internvl_infer. 
#User permissions → Repositories: 
check “Read access to contents of all public gated repos you can access.” Leave all other permissions unchecked (no Write, no Inference permissions needed). 

Click Generate token, then copy and store it securely (you’ll only see it once).
```
