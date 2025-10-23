import json

# 输入和输出文件名
input_file = "raw_dataset.json"  # 原始 JSON 文件
output_file = "grpo_training_dataset.jsonl"  # 目标 JSONL 文件

# 读取原始 JSON 文件
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)  # 原文件是列表格式

# 写入 JSONL 文件
with open(output_file, "w", encoding="utf-8") as f:
    for item in data:
        # 只保留指定字段
        filtered_item = {
            "channel": item["channel"],
            "messages": item["messages"],
            "videos": item["videos"]
        }
        # 每行写入一个 JSON 字典
        f.write(json.dumps(filtered_item, ensure_ascii=False) + "\n")

print(f" 已成功保存为 {output_file}")