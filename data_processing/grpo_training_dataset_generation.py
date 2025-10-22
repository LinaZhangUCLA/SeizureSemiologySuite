import os
import json

# 设定根目录
root_dir = "./ft_data"  # 改成你的路径
merged_data = []

# 遍历所有子文件夹
for folder, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".json"):
            file_path = os.path.join(folder, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # 每个文件可能是一个列表或单个对象
                    if isinstance(data, list):
                        merged_data.extend(data)
                    else:
                        merged_data.append(data)
                print(f"✅ 已读取: {file_path}")
            except Exception as e:
                print(f"⚠️ 读取失败 {file_path}: {e}")

# 输出合并后的文件
output_path = "grpo_training_dataset.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)

print(f"\n🎉 合并完成，共 {len(merged_data)} 条数据，已保存到 {output_path}")