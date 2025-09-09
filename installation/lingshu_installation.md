## Environment Installation (for Lingshu + Qwen2.5-VL)

```bash
# If you already have the qwenvl conda environment, follow these steps to save time.
conda create --name lingshu  --clone qwen25
conda activate lingshu 
conda install -c conda-forge ffmpeg -y 
pip install ffmpeg-python
```

```bash
# 1. Create new conda env
conda create -n lingshu python=3.10 -y
conda activate lingshu

# 2. Install ffmpeg (for video reading)
conda install -c conda-forge ffmpeg -y
pip install ffmpeg-python

# 3. Install PyTorch + key dependencies
# (try cu124 first; if not working on your server, switch cu124 → cu121)
pip install torch torchvision torchaudio   transformers==4.51.3 accelerate qwen-vl-utils pandas peft tqdm numpy scipy   --extra-index-url https://download.pytorch.org/whl/cu124
pip install -U peft
# 4. Install FlashAttention (force rebuild, safest)
pip install --upgrade setuptools wheel
pip uninstall -y flash-attn || true
pip cache purge
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn --no-build-isolation --no-cache-dir
```
