import json
from datetime import datetime
# 输入和输出文件名
# input_file = "../../dataset/sft_merge_2025-10-26_swift.jsonl"  # 原始 JSONL 文件
# output_file = "../../dataset/grpo_merge_2025-10-26.jsonl"  # 目标 JSONL 文件

DEFAULT_DATE = datetime.now().strftime("%Y-%m-%d")
DEFAULT_DATE = '2025-10-28'


input_file = f"../dataset/sft_merge_{DEFAULT_DATE}_swift_train.jsonl"  # 原始 JSONL 文件
output_file = f"../dataset/grpo_merge_{DEFAULT_DATE}_swift_train.jsonl"  # 目标 JSONL 文件

seen_tasks = set()  # 用于记录已经处理过的 task

# 打开输入和输出文件
with (open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout):
    for line in fin:
        if not line.strip():
            continue  # 跳过空行
        item = json.loads(line)

        # # TODO: 测试使用
        # task = item.get("channel", "")
        # if task in seen_tasks:
        #     continue  # 已经处理过这个 task，跳过
        # seen_tasks.add(task)

        # TODO: 修改 messages 的 system 命令，添加 <think>...<think> 和 <answer>...<answer> 的逻辑
        for msg in item["messages"]:
            if msg["role"] == "system":
                msg["content"] = 'You are a medical assistant helping to observe, describe, and analyze seizure videos. When answering, first reason step by step inside <think>...</think> tags, and then give the final answer inside <answer>...</answer>, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.'

        # for msg in item["messages"]:
        #     if msg["role"] == "user":
        #         msg["content"] = 'describe the video.\n\n<video>'



        # 保留指定字段
        filtered_item = {
            "task": item.get("channel", ""),   # 防止有缺失
            "messages": item.get("messages", []),
            "solution": [m["content"] for m in item.get("messages", []) if m.get("role") == "assistant"][0],
            "videos": item["videos"]
            # "videos": ["./video_demo_debug.mp4"]
        }

        # 写入 JSONL
        fout.write(json.dumps(filtered_item, ensure_ascii=False) + "\n")

print(f"✅ 已成功保存为 {output_file}")