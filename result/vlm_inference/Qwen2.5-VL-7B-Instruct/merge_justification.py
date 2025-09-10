import pandas as pd
import re
from collections import OrderedDict

# ====================================
input_csv = "/Users/pxy_amber/Downloads/Task1_Qwen2.5-VL-7B-Instruct_all.csv"
output_csv = "/Users/pxy_amber/Downloads/Task1_Qwen2.5-VL-7B-Instruct_all_merged.csv"
# ====================================

def check_duplecated(df_tmp):
    first_column = df_tmp.iloc[:, 0]
    duplicates = first_column.duplicated()
    if duplicates.any():
        print("repeat files:")
        print(first_column[duplicates].unique())
    else:
        print("No repeat video file.")

def get_original_filename(fn: str) -> str:
    return re.sub(r'_segment_\d+', '', str(fn))

def any_yes(series: pd.Series) -> str:
    vals = series.astype(str).str.strip().str.lower()
    if (vals == "yes").any():
        return "yes"
    if len(vals) > 0 and (vals == "no").all():
        return "no"
    return "N/A"

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

df = pd.read_csv(input_csv)
check_duplecated(df)

if "file_name" not in df.columns:
    raise ValueError("input CSV does not find 'file_name'column.")

df['orig_file_name'] = df['file_name'].apply(get_original_filename)

features = []
cols = list(df.columns)
for c in cols:
    if c in {"file_name", "orig_file_name"}:
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
    merged_row = {"file_name": orig_name}

    for feat in features:
        label_col = feat
        just_col = f"justification_for_{feat}"

        seg_labels = group[label_col].astype(str).str.strip().str.lower()
        seg_justs  = group[just_col].astype(str)

        # 1) video-level label（any-of-yes）
        merged_row[feat] = any_yes(seg_labels)

        # 2) merge justification
        has_yes = (seg_labels == "yes").any()
        has_no  = (seg_labels == "no").any()
        all_yes = len(seg_labels) > 0 and (seg_labels == "yes").all()
        all_no  = len(seg_labels) > 0 and (seg_labels == "no").all()
        mixed   = has_yes and has_no

        if all_yes or all_no:
            selected = seg_justs.tolist()
        elif mixed:
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

result_df.to_csv(output_csv, index=False)
print("Merge complete, result saved to", output_csv)

print("\n[Preview]")
print(result_df.head(5))
