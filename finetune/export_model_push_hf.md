# SFT


```shell
# if use your local model path, add --model parameter, otherwise no need --model.

conda activate swift-ft
cd fintune  # if you use relative path

CUDA_VISIBLE_DEVICES=3 \
swift export \
  --model /home/lina/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-7B \
  --adapters ./ckpts/trained_models/qwen_2.5_omni_task17_grpo_20251031005854/v0-20251031-005919/checkpoint-600 \
  --merge_lora true \
  --to_hf true \
  --output_dir ./sft/seizure_omni_sft


hf upload \
  CedrusLNZ/seizure_omni_sft \
  ./sft/seizure_omni_sft \
  --token $HF_TOKEN \
  --repo-type model
```

# GRPO



```shell
conda activate swift-grpo
cd fintune  # if you use relative path

CUDA_VISIBLE_DEVICES=3 \
swift export \
  --model /home/lina/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-7B \
  --adapters ./ckpts/trained_models/qwen_2.5_omni_task17_grpo_20251031005854/v0-20251031-005919/checkpoint-600 \
  --merge_lora true \
  --to_hf true \
  --output_dir ./rlhf/seizure_omni_grpo


hf upload \
  CedrusLNZ/seizure_omni_grpo \
  ./rlhf/seizure_omni_grpo \
  --token $HF_TOKEN \
  --repo-type model
```