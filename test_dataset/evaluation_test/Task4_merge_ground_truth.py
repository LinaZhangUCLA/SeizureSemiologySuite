import os
import pandas as pd

# 1. 你给的分组
feature_folders_split = [
    ['blank_stare', 'close_eyes', 'eye_blinking'],
    ['tonic', 'clonic', 'arm_flexion'],
    ['arm_straightening', 'figure4', 'oral_automatisms'],
    ['limb_automatisms', 'face_pulling'],
    ['head_turning', 'asynchronous_movement'],
    ['pelvic_thrusting', 'arms_move_simultaneously'],
    ['full_body_shaking', 'start'],
    ['face_twitching', 'end'],
    # 'ictal_vocalization', 'verbal_responsiveness','occur_during_sleep',
]

# 2. 拍平成一个 feature list
features = [f for group in feature_folders_split for f in group]

# 3. ground truth 目录（按你的截图）
GROUND_TRUTH_DIR = "/home/lina/ssb/SeizureSemiologyBench/result/ground_truth/task4_groundtruth"   # 如果路径不一样自己改

all_dfs = []

for feat in features:
    csv_name = f"{feat}_timestamps.csv"   # e.g. arm_flexion_timestamps.csv
    csv_path = os.path.join(GROUND_TRUTH_DIR, csv_name)

    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} 不存在，跳过。")
        continue

    # 读这个 feature 的 csv
    df = pd.read_csv(csv_path)

    # 确保有这两列
    # 如果原文件有多余列也无所谓，它们会一起被合并
    if 'file_name' not in df.columns or 'target_time' not in df.columns:
        print(f"[WARN] {csv_path} 没有 file_name 或 target_time 列，实际列是 {df.columns.tolist()}，先跳过。")
        continue

    # 新增一列 feature
    df['feature'] = feat

    all_dfs.append(df)

# 4. 合并并写出
if all_dfs:
    merged = pd.concat(all_dfs, ignore_index=True)
    output_csv = "/home/lina/ssb/SeizureSemiologyBench/result/ground_truth/Task4_time_annotation.csv"   # 想叫 Tas4_time_annotation.csv 就改这一行
    merged.to_csv(output_csv, index=False)
    print(f"[OK] 合并完成，共 {len(merged)} 行，已写入 {output_csv}")
else:
    print("[ERROR] 没有读到任何 CSV，检查目录和文件名是否正确。")
