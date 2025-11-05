import pandas as pd

# 路径
path_a = "/home/lina/ssb/SeizureSemiologyBench/result/vlm_inference/Qwen3-VL-8B-Instruct/Task12_Qwen3-VL-8B-Instruct_all_merged.csv"
path_b = "/home/lina/ssb/SeizureSemiologyBench/result/vlm_inference/Qwen3-VL-8B-Instruct/Task12_Qwen3-VL-8B-Instruct_all_merged_llmmerge.csv"
path_c = "/home/lina/ssb/SeizureSemiologyBench/result/vlm_inference/Qwen3-VL-8B-Instruct/Task12_Qwen3-VL-8B-Instruct_diff.csv"

# 读入
df_a = pd.read_csv(path_a)
df_b = pd.read_csv(path_b)

# 取出 B 里面已有的 file_name 集合
b_names = set(df_b["file_name"].astype(str))

# 从 A 里挑出那些 file_name 不在 B 里的行
df_diff = df_a[~df_a["file_name"].astype(str).isin(b_names)]

# 存成 C
df_diff.to_csv(path_c, index=False)
print(f"done, saved to {path_c}, rows: {len(df_diff)}")
