import os
import re
import glob
import pandas as pd

# === Set your input folder and output path ===
INPUT_DIR = "/mnt/SSD3/lina/resultsz"
OUTPUT_CSV = "/mnt/SSD3/xinyi/benchmark/output/Task1_Qwen2.5-VL-7B-Instruct_all.csv"
DEDUP_ON_COLUMN = None 
# ============================================

# Match the target CSVs
pattern = os.path.join(INPUT_DIR, "Task1_Qwen2.5-VL-7B-Instruct_*.csv")
files = glob.glob(pattern)

if not files:
    raise FileNotFoundError(f"No matching files found in: {INPUT_DIR}")

def sort_key(path: str):
    """
    Extract numeric start/end indices from filename to sort properly.
    Example: '..._901-1200.csv' -> (901, 1200)
    If extraction fails, fall back to filename-based sorting.
    """
    name = os.path.basename(path)
    m = re.search(r'_(\d+)-(\d+)\.csv$', name)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    # Fallback: extract all numbers or use the name itself
    nums = re.findall(r'\d+', name)
    return tuple(int(x) for x in nums) if nums else (name,)

files_sorted = sorted(files, key=sort_key)
print("Merging in the following order:")
for f in files_sorted:
    print(" -", os.path.basename(f))

# Read and concatenate
dfs = []
for fp in files_sorted:
    df = pd.read_csv(fp, dtype=str)          # use string dtype to avoid type mismatches
    df["__source_file"] = os.path.basename(fp)  # provenance column (optional)
    dfs.append(df)

merged = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

# Optional de-duplication on a specific column
if DEDUP_ON_COLUMN is not None and DEDUP_ON_COLUMN in merged.columns:
    before = len(merged)
    merged = merged.drop_duplicates(subset=[DEDUP_ON_COLUMN], keep="first")
    after = len(merged)
    print(f"De-duplicated on '{DEDUP_ON_COLUMN}': {before} -> {after}")

# Optionally drop the provenance column:
# merged = merged.drop(columns=["__source_file"], errors="ignore")

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Save
merged.to_csv(OUTPUT_CSV, index=False)
print("Merge complete. Saved to:", OUTPUT_CSV)
print("Total rows:", len(merged))
