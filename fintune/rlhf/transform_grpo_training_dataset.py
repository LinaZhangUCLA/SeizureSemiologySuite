import json

# 输入和输出文件名
# input_file = "../../dataset/sft_merge_2025-10-26_swift.jsonl"  # 原始 JSONL 文件
# output_file = "../../dataset/grpo_merge_2025-10-26.jsonl"  # 目标 JSONL 文件

input_file = "sft_merge_2025-10-26_swift.jsonl"  # 原始 JSONL 文件
output_file = "grpo_merge_2025-10-26_swfit.jsonl"  # 目标 JSONL 文件

# 打开输入和输出文件
with (open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout):
    for line in fin:
        if not line.strip():
            continue  # 跳过空行
        item = json.loads(line)

        # TODO: 修改 messages 的 system 命令，添加 <think>...<think> 和 <answer>...<answer> 的逻辑
        for msg in item["messages"]:
            if msg["role"] == "system":
                msg["content"] = 'You are a medical assistant helping to observe, describe, and analyze seizure videos. When answering, first reason step by step inside <think>...</think> tags, and then give the final answer inside <answer>...</answer>, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.'

        # 保留指定字段
        filtered_item = {
            "task": item.get("channel", ""),   # 防止有缺失
            "messages": item.get("messages", []),
            "videos": item["videos"]
            # "videos": "./video_demo.mp4"
        }

        # 写入 JSONL
        fout.write(json.dumps(filtered_item, ensure_ascii=False) + "\n")

print(f"✅ 已成功保存为 {output_file}")