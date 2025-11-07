#!/bin/bash

# Run SeizureRQI evaluation on VLM outputs

# Set your OpenAI API key (or export it as environment variable)
# export OPENAI_API_KEY="your-api-key-here"

# Define paths
VLM_CSV="./result/vlm_inference/Qwen2.5-VL-32B-Instruct/Task6_Qwen2.5-VL-32B-Instruct_all_merged_llmmerge.csv"
GT_CSV="./result/ground_truth/task6_annotation.csv"
OUTPUT_DIR="./metrics/SeizureRQI"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run evaluation for Qwen2.5-VL-32B
echo "Evaluating Qwen2.5-VL-32B-Instruct..."
python evaluation/seizure_rqi_evaluation.py \
    --vlm_csv "$VLM_CSV" \
    --gt_csv "$GT_CSV" \
    --output "$OUTPUT_DIR/seizure_rqi_Qwen2.5-VL-32B.csv" \
    --model "gpt-4o"

echo "Evaluation complete! Results saved in $OUTPUT_DIR"