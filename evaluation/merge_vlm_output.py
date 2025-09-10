import os
import re
import glob
import pandas as pd
import csv
import shutil


# === Set your input folder and output path ===
BASE_DIR = "/mnt/SSD3/lina/ssb/"
DEDUP_ON_COLUMN = 'file_name' 
# ============================================

def read_csv_file(input_file: str) -> pd.DataFrame:
    corrected_data = []
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)     
        header = next(reader)      
        for row in reader:
            if len(row) > 3:
                first_two_cols = row[:2]
                merged_col = ','.join(row[2:])
                corrected_row = first_two_cols + [merged_col]
                corrected_data.append(corrected_row)
            else:
                corrected_data.append(row)
    df = pd.DataFrame(corrected_data, columns=header)
    return df

def mergecsv(task_name: str, model: str,subtask:str=None):
    INPUT_DIR = BASE_DIR +"vlm/" + model
    # Match the target CSVs
    pattern = os.path.join(INPUT_DIR, f"{task_name}_{model}_*.csv")
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
    header = []
    for fp in files_sorted:
        print(fp)
        if task_name=='Task3_6' :
            df = read_csv_file(fp)
        else:
            df = pd.read_csv(fp, dtype=str)          # use string dtype to avoid type mismatches
        #df["__source_file"] = os.path.basename(fp)  # provenance column (optional)
        if not header:
            header = df.columns.tolist()
        else:
            # Reorder columns to match the first file
            if list(df.columns) != header:
                print(f"Warning: Column mismatch in {os.path.basename(fp)}. Reordering columns. The first file header is {header}, but this file has {list(df.columns)}")
                raise ValueError("Column mismatch detected.")
                df = df.reindex(columns=header)
        dfs.append(df)

    merged = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

    if subtask=='Task3':
        merged = merged[["video_name", "event_sequence"]]
        merged = merged.rename(columns={"video_name": "file_name"})
    if subtask=='Task6':    
        merged = merged[["video_name", "report"]]
        merged = merged.rename(columns={"video_name": "file_name"})
    if subtask=='Task4L':    
        merged = merged[["video_name", "onset_body_part"]]
        merged = merged.rename(columns={"video_name": "file_name"})
    if subtask=='Task5':    
        merged = merged[["video_name", "start_time","end_time"]]
        merged = merged.rename(columns={"video_name": "file_name"})    

    # Optional de-duplication on a specific column
    if DEDUP_ON_COLUMN is not None and DEDUP_ON_COLUMN in merged.columns:
        before = len(merged)
        dup_rows = merged[merged.duplicated(subset=[DEDUP_ON_COLUMN], keep=False)]
        if not dup_rows.empty:
            print("Duplicate rows before dropping:")
            print(dup_rows)
        merged = merged.drop_duplicates(subset=[DEDUP_ON_COLUMN], keep="first")
        after = len(merged)
        print(f"De-duplicated on '{DEDUP_ON_COLUMN}': {before} -> {after}")

    # Optionally drop the provenance column:
    # merged = merged.drop(columns=["__source_file"], errors="ignore")

    # Ensure output directory exists
    # Save
    if not subtask:
        subtask = task_name
    OUTPUT_CSV = BASE_DIR + model + f"/{subtask}_{model}_all.csv"
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    merged.to_csv(OUTPUT_CSV, index=False)
    print("Merge complete. Saved to:", OUTPUT_CSV)
    print("Total rows:", len(merged))


if __name__ == "__main__":
    
    model = 'Qwen2.5-VL-7B-Instruct'
    task_name = 'Task1'
    mergecsv(task_name, model)      
    task_name = 'Task3_6'
    subtask = 'Task3'
    mergecsv(task_name, model,subtask)
    subtask = 'Task6'
    mergecsv(task_name, model,subtask)
    shutil.copy(BASE_DIR + "vlm/" + model + "/Task4_AM_Qwen2.5-VL-7B-Instruct_1-112.csv", BASE_DIR + model + "/Task4_AM_Qwen2.5-VL-7B-Instruct_1-112.csv")
    shutil.copy(BASE_DIR + "vlm/" + model + "/Task4_HT_Qwen2.5-VL-7B-Instruct_1-129.csv", BASE_DIR + model + "/Task4_HT_Qwen2.5-VL-7B-Instruct_1-129.csv")
    task_name = 'Task4L_5'
    subtask = 'Task4L'
    mergecsv(task_name, model,subtask)
    subtask = 'Task5'
    mergecsv(task_name, model,subtask)