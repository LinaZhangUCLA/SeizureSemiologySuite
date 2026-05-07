import re
import os
import pandas as pd
from typing import List


def parse_event_sequence(value):
    """
    支持以下格式：
    - 'a,b,c' 或 'a，b，c'（英文/中文逗号）
    - 'a, b, c'（含空格）
    - 类 JSON / Python 列表字符串："['a','b','c']" 或 ["a","b","c"]（用正则提取引号中的项）
    若为空则返回空列表。
    """
    if pd.isna(value):
        return []
    s = str(value).strip()

    # 处理形如 [ ... ] 的列表字符串
    if s.startswith('[') and s.endswith(']'):
        inner = s[1:-1].strip()
        # 先提取引号中的条目（同时支持单/双引号）
        tokens = re.findall(r"""['"]([^'"]+)['"]""", inner)
        if tokens:
            return [t.strip() for t in tokens if t.strip() != ""]
        # 若没有引号，就按中英文逗号分割
        parts = re.split(r'[，,]', inner)
        return [p.strip(" '\"") for p in parts if p.strip(" '\"") != ""]

    # 常规：按中英文逗号分割
    if ('，' in s) or (',' in s):
        parts = re.split(r'[，,]', s)
        return [p.strip() for p in parts if p.strip() != ""]

    # 兜底：按空白分割
    return [p.strip() for p in s.split() if p.strip() != ""]

def maximal_overlap_merge(seq_a: List[str], seq_b: List[str]) -> List[str]:
    """
    最大后缀/前缀重叠拼接：
    例如 [a,b,c] 与 [b,c,d] -> [a,b,c,d]
    无重叠则直接相连。
    """
    if not seq_a:
        return list(seq_b)
    if not seq_b:
        return list(seq_a)
    max_k = min(len(seq_a), len(seq_b))
    overlap = 0
    for k in range(max_k, 0, -1):
        if seq_a[-k:] == seq_b[:k]:
            overlap = k
            break
    return seq_a + seq_b[overlap:]

def merge_all(seqs: List[List[str]]) -> List[str]:
    merged = []
    for seq in seqs:
        merged = maximal_overlap_merge(merged, seq)
    return merged

def process_csv(input_csv_path: str) -> str:
    """
    - 从 file_name 提取 _segment_<数字> 为 segment_num
    - 去掉 “_segment_<数字>” 得到 original_file_name
    - 分组后按 segment_num 升序，将 event_sequence 用最大重叠方式合并为 feature_list
    - 输出两列：original_file_name, feature_list
    - 保存到同目录，文件名为 'merge' + 原文件名
    """
    df = pd.read_csv(input_csv_path)
    if 'file_name' not in df.columns or 'event_sequence' not in df.columns:
        raise ValueError("CSV必须包含列: file_name, event_sequence")

    def extract_segment_num(fn: str) -> int:
        if pd.isna(fn):
            return -1
        m = re.search(r'_segment_(\d+)', str(fn))
        return int(m.group(1)) if m else -1

    def strip_segment_tag(fn: str) -> str:
        if pd.isna(fn):
            return ""
        return re.sub(r'_segment_\d+', '', str(fn))

    df['segment_num'] = df['file_name'].apply(extract_segment_num)
    df['original_file_name'] = df['file_name'].apply(strip_segment_tag)

    out_rows = []
    for orig, sub in df.groupby('original_file_name', sort=False):
        sub_sorted = sub.sort_values(by='segment_num', kind='mergesort')
        seqs = [parse_event_sequence(x) for x in sub_sorted['event_sequence'].tolist()]
        merged_seq = merge_all(seqs)
        feature_list = ",".join(merged_seq)
        out_rows.append({"file_name": orig, "feature_list": feature_list})

    out_df = pd.DataFrame(out_rows, columns=["file_name", "feature_list"])

    
    root, ext = os.path.splitext(input_csv_path)   # ext == ".csv"
    out_path = root + "_merge" + ext

    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path

if __name__ == "__main__":

    # input_csv = "/home/lina/ssb/SeizureSemiologyBench/result/vlm_inference/Qwen2.5-VL-7B-Instruct/task5.csv"
    # out_file = process_csv(input_csv)
    # print(f"已生成: {out_file}") 
    # Model names
    model_names = [
        # 'InternVL3_5-8B',
        # 'Qwen2.5-VL-7B-Instruct',
        # 'Qwen3-VL-8B-Instruct',
        # 'InternVL3_5-38B',
        # 'Qwen2.5-VL-32B-Instruct',
        # 'Qwen3-VL-32B-Instruct',
        # 'Qwen2.5-VL-72B-Instruct',
        # 'Qwen2.5-Omni-7B',
        # "Qwen3-Omni-30B-A3B-Instruct",
        # 'Lingshu-32B',    
        'seizure_omni_sft' ,
        'seizure_omni_grpo'   
    ]
    
    for model in model_names:
        print(model)
        base_dir = '/home/lina/ssb/SeizureSemiologyBench/result/vlm_inference_test/'
        input_csv = f"{base_dir}{model}/Task5_{model}_all.csv"
        if os.path.isfile(input_csv):
            out_file = process_csv(input_csv)
            print(f"已生成: {out_file}")
    

    
  