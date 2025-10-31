#merge all.csv to all_merge_csv
import os
import pandas as pd
import re
from collections import OrderedDict

# ====================================
BASE_DIR = "result/vlm_inference"
MODELS = [
    # "InternVL3_5-8B",
    # "InternVL3_5-38B",
    # "Qwen2.5-VL-7B-Instruct",
    # "Qwen2.5-VL-32B-Instruct",
    # "Qwen2.5-VL-72B-Instruct",
    # "Lingshu-32B",
    # "Qwen2.5-Omni-7B",
    'Qwen3-VL-8B-Instruct',
    'Qwen3-VL-32B-Instruct'
]
# ====================================

def check_duplecated(df_tmp):
    target_col = None
    for c in ("file_name", "video_name"):
        if c in df_tmp.columns:
            target_col = c
            break
    if target_col is None:
        print('No "file_name" or "video_name" column found.')
        return
    s = df_tmp[target_col].astype(str).str.strip()
    s = s[s != ""]
    dup_mask = s.duplicated()
    if dup_mask.any():
        print(f'repeat files (column: "{target_col}"):')
        print(s[dup_mask].unique())
    else:
        print("No repeat video file.")

def get_original_filename(fn: str) -> str:
    return re.sub(r'_segment_\d+', '', str(fn))

def get_segment_index(fn: str) -> int:
    s = str(fn)
    m = re.search(r'_segment_(\d+)', s)
    if m:
        return int(m.group(1))
    m = re.search(r'_(\d+)(?=\.\w+$)', s)
    if m:
        return int(m.group(1))
    return 10**9

def any_yes(series: pd.Series) -> str:
    vals = series.astype(str).str.strip().str.lower()
    if (vals == "yes").any():
        return "yes"
    else:
        return "no"

def dedup_join(texts) -> str:
    seen = OrderedDict()
    for t in texts:
        if pd.isna(t):
            continue
        s = str(t).strip()
        if not s:
            continue
        if s not in seen:
            seen[s] = True
    return " || ".join(seen.keys())

def merge_one_csv(input_csv: str, output_csv: str):
    print(f"\n=== Merging: {os.path.basename(input_csv)} ===")

    df = pd.read_csv(input_csv)
    check_duplecated(df)

    # 允许 file_name 或 video_name
    base_col = "file_name" if "file_name" in df.columns else ("video_name" if "video_name" in df.columns else None)
    if base_col is None:
        raise ValueError('input CSV must contain "file_name" or "video_name".')

    df['orig_file_name'] = df[base_col].apply(get_original_filename)
    df['_seg_index'] = df[base_col].apply(get_segment_index)

    features = []
    cols = list(df.columns)
    skip_cols = {base_col, "file_name", "video_name", "orig_file_name", "_seg_index"}
    for c in cols:
        if c in skip_cols:
            continue
        if c.startswith("justification_for_"):
            continue
        just_col = f"justification_for_{c}"
        if just_col in df.columns:
            features.append(c)

    if not features:
        raise ValueError("not find <feature> / justification_for_<feature> column.")

    result_rows = []

    for orig_name, group in df.groupby('orig_file_name', sort=False):
        group_sorted = group.sort_values(by=['_seg_index'], kind='mergesort')
        merged_row = {"file_name": orig_name}

        for feat in features:
            label_col = feat
            just_col  = f"justification_for_{feat}"

            seg_labels = group_sorted[label_col].astype(str).str.strip().str.lower()
            seg_justs  = group_sorted[just_col].astype(str)

            video_label = any_yes(seg_labels)
            merged_row[feat] = video_label

            has_yes = (seg_labels == "yes").any()
            has_no  = (seg_labels == "no").any()
            all_yes = len(seg_labels) > 0 and (seg_labels == "yes").all()
            all_no  = len(seg_labels) > 0 and (seg_labels == "no").all()
            mixed   = has_yes and has_no

            if feat == "occur_during_sleep":
                first_just = str(seg_justs.iloc[0]).strip() if len(seg_justs) > 0 else ""
                merged_row[just_col] = first_just
            else:
                if all_yes or all_no:
                    selected = seg_justs.tolist()
                elif mixed:
                    # 仅合并 yes 段
                    selected = [j for l, j in zip(seg_labels.tolist(), seg_justs.tolist()) if l == "yes"]
                else:
                    selected = seg_justs.tolist()

                merged_row[just_col] = dedup_join(selected)

        result_rows.append(merged_row)

    ordered_cols = ["file_name"]
    for feat in features:
        ordered_cols.append(feat)
        ordered_cols.append(f"justification_for_{feat}")

    result_df = pd.DataFrame(result_rows)[ordered_cols]

    check_duplecated(result_df)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result_df.to_csv(output_csv, index=False)
    print("Merge complete, result saved to", output_csv)

    print("\n[Preview]")
    print(result_df.head(5))

def main():
    for model in MODELS:
        # 构建输入和输出路径：都在对应的模型目录下
        model_dir = os.path.join(BASE_DIR, model)
        in_csv  = os.path.join(model_dir, f"Task1_{model}_all.csv")
        out_csv = os.path.join(model_dir, f"Task12_{model}_all_merged.csv")
        
        if not os.path.exists(in_csv):
            print(f"[Skip] Input not found: {in_csv}")
            continue
        try:
            merge_one_csv(in_csv, out_csv)
        except Exception as e:
            print(f"[Error] {model}: {e}")

if __name__ == "__main__":
    main()




#previous
# import os
# import pandas as pd
# import re
# from collections import OrderedDict

# # ====================================
# BASE_DIR = "/Users/pxy_amber/Downloads"
# MODELS = [
#     "InternVL3_5-8B",
#     "InternVL3_5-38B",
#     "Qwen2.5-VL-7B-Instruct",
#     "Qwen2.5-VL-32B-Instruct",
#     "Qwen2.5-VL-72B-Instruct",
#     "Lingshu-32B",
#     "Qwen2.5-Omni-7B",
#     'Qwen3-VL-8B-Instruct',
#     'Qwen3-VL-32B-Instruct'

# ]
# # ====================================

# def check_duplecated(df_tmp):
#     target_col = None
#     for c in ("file_name", "video_name"):
#         if c in df_tmp.columns:
#             target_col = c
#             break
#     if target_col is None:
#         print('No "file_name" or "video_name" column found.')
#         return
#     s = df_tmp[target_col].astype(str).str.strip()
#     s = s[s != ""]
#     dup_mask = s.duplicated()
#     if dup_mask.any():
#         print(f'repeat files (column: "{target_col}"):')
#         print(s[dup_mask].unique())
#     else:
#         print("No repeat video file.")

# def get_original_filename(fn: str) -> str:
#     return re.sub(r'_segment_\d+', '', str(fn))

# def get_segment_index(fn: str) -> int:
#     s = str(fn)
#     m = re.search(r'_segment_(\d+)', s)
#     if m:
#         return int(m.group(1))
#     m = re.search(r'_(\d+)(?=\.\w+$)', s)
#     if m:
#         return int(m.group(1))
#     return 10**9

# def any_yes(series: pd.Series) -> str:
#     vals = series.astype(str).str.strip().str.lower()
#     if (vals == "yes").any():
#         return "yes"
#     else:
#         return "no"

# def dedup_join(texts) -> str:
#     seen = OrderedDict()
#     for t in texts:
#         if pd.isna(t):
#             continue
#         s = str(t).strip()
#         if not s:
#             continue
#         if s not in seen:
#             seen[s] = True
#     return " || ".join(seen.keys())

# def merge_one_csv(input_csv: str, output_csv: str):
#     print(f"\n=== Merging: {os.path.basename(input_csv)} ===")

#     df = pd.read_csv(input_csv)
#     check_duplecated(df)

#     # 允许 file_name 或 video_name
#     base_col = "file_name" if "file_name" in df.columns else ("video_name" if "video_name" in df.columns else None)
#     if base_col is None:
#         raise ValueError('input CSV must contain "file_name" or "video_name".')

#     df['orig_file_name'] = df[base_col].apply(get_original_filename)
#     df['_seg_index'] = df[base_col].apply(get_segment_index)

#     features = []
#     cols = list(df.columns)
#     skip_cols = {base_col, "file_name", "video_name", "orig_file_name", "_seg_index"}
#     for c in cols:
#         if c in skip_cols:
#             continue
#         if c.startswith("justification_for_"):
#             continue
#         just_col = f"justification_for_{c}"
#         if just_col in df.columns:
#             features.append(c)

#     if not features:
#         raise ValueError("not find <feature> / justification_for_<feature> column.")

#     result_rows = []

#     for orig_name, group in df.groupby('orig_file_name', sort=False):
#         group_sorted = group.sort_values(by=['_seg_index'], kind='mergesort')
#         merged_row = {"file_name": orig_name}

#         for feat in features:
#             label_col = feat
#             just_col  = f"justification_for_{feat}"

#             seg_labels = group_sorted[label_col].astype(str).str.strip().str.lower()
#             seg_justs  = group_sorted[just_col].astype(str)

#             video_label = any_yes(seg_labels)
#             merged_row[feat] = video_label

#             has_yes = (seg_labels == "yes").any()
#             has_no  = (seg_labels == "no").any()
#             all_yes = len(seg_labels) > 0 and (seg_labels == "yes").all()
#             all_no  = len(seg_labels) > 0 and (seg_labels == "no").all()
#             mixed   = has_yes and has_no

#             if feat == "occur_during_sleep":
#                 first_just = str(seg_justs.iloc[0]).strip() if len(seg_justs) > 0 else ""
#                 merged_row[just_col] = first_just
#             else:
#                 if all_yes or all_no:
#                     selected = seg_justs.tolist()
#                 elif mixed:
#                     # 仅合并 yes 段
#                     selected = [j for l, j in zip(seg_labels.tolist(), seg_justs.tolist()) if l == "yes"]
#                 else:
#                     selected = seg_justs.tolist()

#                 merged_row[just_col] = dedup_join(selected)

#         result_rows.append(merged_row)

#     ordered_cols = ["file_name"]
#     for feat in features:
#         ordered_cols.append(feat)
#         ordered_cols.append(f"justification_for_{feat}")

#     result_df = pd.DataFrame(result_rows)[ordered_cols]

#     check_duplecated(result_df)

#     os.makedirs(os.path.dirname(output_csv), exist_ok=True)
#     result_df.to_csv(output_csv, index=False)
#     print("Merge complete, result saved to", output_csv)

#     print("\n[Preview]")
#     print(result_df.head(5))

# def main():
#     for model in MODELS:
#         in_csv  = os.path.join(BASE_DIR, f"Task1_{model}_all.csv")
#         out_csv = os.path.join(BASE_DIR, f"Task1_{model}_all_merged.csv")
#         if not os.path.exists(in_csv):
#             print(f"[Skip] Input not found: {in_csv}")
#             continue
#         try:
#             merge_one_csv(in_csv, out_csv)
#         except Exception as e:
#             print(f"[Error] {model}: {e}")

# if __name__ == "__main__":
#     main()
