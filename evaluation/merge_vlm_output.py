import os
import re
import glob
import pandas as pd
import csv
import shutil


# === Set your input folder and output path ===
BASE_DIR = ""
DEDUP_ON_COLUMN = 'file_name' 
origin_dir = BASE_DIR + "inference_result/"
output_dir = BASE_DIR + "result/vlm_inference/"
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

def detect_available_tasks(input_dir: str, model: str):
    """
    检测input_dir下有哪些task文件
    返回：{'Task1': True, 'Task3_6': True, 'Task4L_5': True, 'Task4_AM': True, 'Task4_HT': True}
    """
    all_files = glob.glob(os.path.join(input_dir, f"*_{model}_*.csv"))
    
    tasks = {
        'Task1': False,
        'Task3_6': False,
        'Task4L_5': False,
        'Task4_AM': False,
        'Task4_HT': False
    }
    
    for f in all_files:
        basename = os.path.basename(f)
        if basename.startswith('Task1_'):
            tasks['Task1'] = True
        elif basename.startswith('Task3_6_'):
            tasks['Task3_6'] = True
        elif basename.startswith('Task4L_5_'):
            tasks['Task4L_5'] = True
        elif basename.startswith('Task4_AM_'):
            tasks['Task4_AM'] = True
        elif basename.startswith('Task4_HT_'):
            tasks['Task4_HT'] = True
    
    return tasks

def mergecsv(task_name: str, model: str, subtask: str = None):
    INPUT_DIR = origin_dir
    # Match the target CSVs
    print(f"{task_name}_{model}_*.csv")
    pattern = os.path.join(INPUT_DIR, f"{task_name}_{model}_*.csv")
    files = glob.glob(pattern)

    if not files:
        print(f"Warning: No matching files found for {task_name}_{model}, skipping...")
        return

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
        if task_name == 'Task3_6':
            df = read_csv_file(fp)
        else:
            df = pd.read_csv(fp, dtype=str)
        if not header:
            header = df.columns.tolist()
        else:
            # Reorder columns to match the first file
            if list(df.columns) != header:
                print(f"Warning: Column mismatch in {os.path.basename(fp)}. Reordering columns. The first file header is {header}, but this file has {list(df.columns)}")
                raise ValueError("Column mismatch detected.")
        dfs.append(df)

    merged = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

    if subtask == 'Task3':
        merged = merged[["video_name", "event_sequence"]]
        merged = merged.rename(columns={"video_name": "file_name"})
    if subtask == 'Task6':    
        merged = merged[["video_name", "report"]]
        merged = merged.rename(columns={"video_name": "file_name"})
    if subtask == 'Task4L':    
        merged = merged[["video_name", "onset_body_part"]]
        merged = merged.rename(columns={"video_name": "file_name"})
    if subtask == 'Task5':    
        merged = merged[["video_name", "start_time", "end_time"]]
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

    # Save
    if not subtask:
        subtask = task_name
    OUTPUT_CSV = output_dir + model + f"/{subtask}_{model}_all.csv"
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    merged.to_csv(OUTPUT_CSV, index=False)
    print("Merge complete. Saved to:", OUTPUT_CSV)
    print("Total rows:", len(merged))
    
    error_log = output_dir + f'row_error_file_stat.log'
    if not os.path.exists(error_log):
        with open(error_log, 'w') as f:
            f.write("Row count errors:\n")
    print(error_log)
    if subtask == 'Task1':
        if len(merged) != 2314:
            with open(error_log, "a") as f:
                f.write(f"{OUTPUT_CSV} has {len(merged)} rows, expected 2314\n")
                print(f"!!!!!!!!!!!!{OUTPUT_CSV} has {len(merged)} rows, expected 2314\n")
    if subtask == 'Task3' or subtask == 'Task6':
        if len(merged) != 2316:
            with open(error_log, "a") as f:
                f.write(f"{OUTPUT_CSV} has {len(merged)} rows, expected 2316\n")
                print(f"!!!!!!!!!!!!{OUTPUT_CSV} has {len(merged)} rows, expected 2316\n")
    if subtask == 'Task5' or subtask == 'Task4L':
        if len(merged) != 2413:
            with open(error_log, "a") as f:
                f.write(f"{OUTPUT_CSV} has {len(merged)} rows, expected 2413\n")
                print(f"!!!!!!!!!!!!{OUTPUT_CSV} has {len(merged)} rows, expected 2413\n")


if __name__ == "__main__":
    # 先检测有哪些模型
    all_files = glob.glob(os.path.join(origin_dir, "*.csv"))
    models = set()
    for f in all_files:
        basename = os.path.basename(f)
        # 提取模型名，假设格式是 TaskX_ModelName_数字.csv
        match = re.search(r'Task\d+(?:_\d+|_[A-Z]+)?_(.+?)_\d+', basename)
        if match:
            models.add(match.group(1))
    
    print(f"Detected models: {models}")
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Processing model: {model}")
        print(f"{'='*60}")
        
        output_modle_dir = output_dir + model
        os.makedirs(output_modle_dir, exist_ok=True)

        # 检测该模型有哪些task
        available_tasks = detect_available_tasks(origin_dir, model)
        print(f"Available tasks for {model}: {available_tasks}")

        # Task1
        if available_tasks['Task1']:
            print(f"\n--- Processing Task1 ---")
            task_name = 'Task1'
            subtask = 'Task1'
            mergecsv(task_name, model, subtask)

        # Task3_6
        if available_tasks['Task3_6']:
            print(f"\n--- Processing Task3_6 ---")
            task_name = 'Task3_6'
            subtask = 'Task3'
            mergecsv(task_name, model, subtask)
            subtask = 'Task6'
            mergecsv(task_name, model, subtask)

        # Task4 AM & HT (直接复制)
        am_end = 112
        ht_end = 129
        if model in ['InternVL3_5-8B', 'InternVL3_5-38B']:
            am_end = 113
            ht_end = 130
        
        if available_tasks['Task4_AM']:
            print(f"\n--- Processing Task4_AM ---")
            src = origin_dir + f"Task4_AM_{model}_1-{am_end}.csv"
            dst = output_dir + model + "/Task4_AM_Qwen2.5-VL-7B-Instruct_1-112.csv"
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"Copied: {src} -> {dst}")
            else:
                print(f"Warning: {src} not found")
        
        if available_tasks['Task4_HT']:
            print(f"\n--- Processing Task4_HT ---")
            src = origin_dir + f"Task4_HT_{model}_1-{ht_end}.csv"
            dst = output_dir + model + "/Task4_HT_Qwen2.5-VL-7B-Instruct_1-129.csv"
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"Copied: {src} -> {dst}")
            else:
                print(f"Warning: {src} not found")

        # Task4L_5
        if available_tasks['Task4L_5']:
            print(f"\n--- Processing Task4L_5 ---")
            task_name = 'Task4L_5'
            subtask = 'Task4L'
            mergecsv(task_name, model, subtask)
            subtask = 'Task5'
            mergecsv(task_name, model, subtask)




#previous
# import os
# import re
# import glob
# import pandas as pd
# import csv
# import shutil


# # === Set your input folder and output path ===
# BASE_DIR = "/mnt/SSD3/lina/ssb/"
# DEDUP_ON_COLUMN = 'file_name' 
# #origin_dir = BASE_DIR + "vlm_original/"
# origin_dir = "/mnt/SSD3/lina/ssb/v3/output/"
# output_dir = BASE_DIR + "v3/vlm_inference/"
# # ============================================

# def read_csv_file(input_file: str) -> pd.DataFrame:
#     corrected_data = []
#     with open(input_file, 'r', newline='', encoding='utf-8') as infile:
#         reader = csv.reader(infile)     
#         header = next(reader)      
#         for row in reader:
#             if len(row) > 3:
#                 first_two_cols = row[:2]
#                 merged_col = ','.join(row[2:])
#                 corrected_row = first_two_cols + [merged_col]
#                 corrected_data.append(corrected_row)
#             else:
#                 corrected_data.append(row)
#     df = pd.DataFrame(corrected_data, columns=header)
#     return df

# def mergecsv(task_name: str, model: str,subtask:str=None):
#     INPUT_DIR = origin_dir #+ model
#     # Match the target CSVs
#     print(f"{task_name}_{model}_*.csv")
#     pattern = os.path.join(INPUT_DIR, f"{task_name}_{model}_*.csv")
#     files = glob.glob(pattern)

#     if not files:
#         raise FileNotFoundError(f"No matching files found in: {INPUT_DIR}")

#     def sort_key(path: str):
#         """
#         Extract numeric start/end indices from filename to sort properly.
#         Example: '..._901-1200.csv' -> (901, 1200)
#         If extraction fails, fall back to filename-based sorting.
#         """
#         name = os.path.basename(path)
#         m = re.search(r'_(\d+)-(\d+)\.csv$', name)
#         if m:
#             return (int(m.group(1)), int(m.group(2)))
#         # Fallback: extract all numbers or use the name itself
#         nums = re.findall(r'\d+', name)
#         return tuple(int(x) for x in nums) if nums else (name,)

#     files_sorted = sorted(files, key=sort_key)
#     print("Merging in the following order:")
#     for f in files_sorted:
#         print(" -", os.path.basename(f))

#     # Read and concatenate
#     dfs = []
#     header = []
#     for fp in files_sorted:
#         print(fp)
#         if task_name=='Task3_6' :
#             df = read_csv_file(fp)
#         else:
#             df = pd.read_csv(fp, dtype=str)          # use string dtype to avoid type mismatches
#         #df["__source_file"] = os.path.basename(fp)  # provenance column (optional)
#         if not header:
#             header = df.columns.tolist()
#         else:
#             # Reorder columns to match the first file
#             if list(df.columns) != header:
#                 print(f"Warning: Column mismatch in {os.path.basename(fp)}. Reordering columns. The first file header is {header}, but this file has {list(df.columns)}")
#                 raise ValueError("Column mismatch detected.")
#                 df = df.reindex(columns=header)
#         dfs.append(df)

#     merged = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

#     if subtask=='Task3':
#         merged = merged[["video_name", "event_sequence"]]
#         merged = merged.rename(columns={"video_name": "file_name"})
#     if subtask=='Task6':    
#         merged = merged[["video_name", "report"]]
#         merged = merged.rename(columns={"video_name": "file_name"})
#     if subtask=='Task4L':    
#         merged = merged[["video_name", "onset_body_part"]]
#         merged = merged.rename(columns={"video_name": "file_name"})
#     if subtask=='Task5':    
#         merged = merged[["video_name", "start_time","end_time"]]
#         merged = merged.rename(columns={"video_name": "file_name"})    

#     # Optional de-duplication on a specific column
#     if DEDUP_ON_COLUMN is not None and DEDUP_ON_COLUMN in merged.columns:
#         before = len(merged)
#         dup_rows = merged[merged.duplicated(subset=[DEDUP_ON_COLUMN], keep=False)]
#         if not dup_rows.empty:
#             print("Duplicate rows before dropping:")
#             print(dup_rows)
#         merged = merged.drop_duplicates(subset=[DEDUP_ON_COLUMN], keep="first")
#         after = len(merged)
#         print(f"De-duplicated on '{DEDUP_ON_COLUMN}': {before} -> {after}")

#     # Optionally drop the provenance column:
#     # merged = merged.drop(columns=["__source_file"], errors="ignore")

#     # Ensure output directory exists
#     # Save
#     if not subtask:
#         subtask = task_name
#     OUTPUT_CSV = output_dir + model + f"/{subtask}_{model}_all.csv"
#     os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
#     merged.to_csv(OUTPUT_CSV, index=False)
#     print("Merge complete. Saved to:", OUTPUT_CSV)
#     print("Total rows:", len(merged))
    
#     error_log = output_dir + f'row_error_file_stat.log'
#     if not os.path.exists(error_log):
#         with open(error_log, 'w') as f:
#             f.write("Row count errors:\n")
#     print(error_log)
#     if subtask == 'Task1':
#        if len(merged) != 2314:
#            #write to row_error.log
#               with open(error_log, "a") as f:
#                   f.write(f"{OUTPUT_CSV} has {len(merged)} rows, expected 2314\n")
#                   print(f"!!!!!!!!!!!!{OUTPUT_CSV} has {len(merged)} rows, expected 2314\n")
#     if subtask == 'Task3' or subtask == 'Task6':
#        if len(merged) != 2316:
#            #write to row_error.log
#               with open(error_log, "a") as f:
#                   f.write(f"{OUTPUT_CSV} has {len(merged)} rows, expected 2316\n")
#                   print(f"!!!!!!!!!!!!{OUTPUT_CSV} has {len(merged)} rows, expected 2316\n")
#     if subtask == 'Task5' or subtask == 'Task4L':
#        if len(merged) != 2413:
#            #write to row_error.log
#               with open(error_log, "a") as f:
#                   f.write(f"{OUTPUT_CSV} has {len(merged)} rows, expected 2413\n")
#                   print(f"!!!!!!!!!!!!{OUTPUT_CSV} has {len(merged)} rows, expected 2413\n")    




# if __name__ == "__main__":

#     for model in ['InternVL3_5-8B','InternVL3_5-38B','Qwen2.5-VL-7B-Instruct','Qwen2.5-VL-32B-Instruct','Qwen2.5-VL-72B-Instruct', 'Lingshu-32B','Qwen2.5-Omni']:
        
#         output_modle_dir = output_dir + model
#         os.makedirs(output_modle_dir, exist_ok=True)

#         #model = 'Qwen2.5-VL-7B-Instruct'
#         # task_name = 'Task1'
#         # subtask = 'Task1'
#         # mergecsv(task_name, model,subtask)      
#         task_name = 'Task3_6'
#         subtask = 'Task3'
#         mergecsv(task_name, model,subtask)
#         subtask = 'Task6'
#         mergecsv(task_name, model,subtask)
#         am_end =112
#         ht_end =129
#         if model in['InternVL3_5-8B','InternVL3_5-38B']:
#             am_end =113
#             ht_end =130
#         # shutil.copy(origin_dir+ model + f"/Task4_AM_{model}_1-{am_end}.csv", output_dir + model + "/Task4_AM_Qwen2.5-VL-7B-Instruct_1-112.csv")
#         # shutil.copy(origin_dir + model + f"/Task4_HT_{model}_1-{ht_end}.csv", output_dir + model + "/Task4_HT_Qwen2.5-VL-7B-Instruct_1-129.csv")
        
#         shutil.copy(origin_dir + f"/Task4_AM_{model}_1-{am_end}.csv", output_dir + model + "/Task4_AM_Qwen2.5-VL-7B-Instruct_1-112.csv")
#         shutil.copy(origin_dir  + f"/Task4_HT_{model}_1-{ht_end}.csv", output_dir + model + "/Task4_HT_Qwen2.5-VL-7B-Instruct_1-129.csv")
       
#         task_name = 'Task4L_5'
#         subtask = 'Task4L'
#         mergecsv(task_name, model,subtask)
#         subtask = 'Task5'
#         mergecsv(task_name, model,subtask)

#     # model = 'InternVL3_5-8B'
#     # task_name = 'Task1'
#     # mergecsv(task_name, model)      
#     # task_name = 'Task3_6'
#     # subtask = 'Task3'
#     # mergecsv(task_name, model,subtask)
#     # subtask = 'Task6'
#     # mergecsv(task_name, model,subtask)
#     # shutil.copy(BASE_DIR + "vlm/" + model + "/Task4_AM_Qwen2.5-VL-7B-Instruct_1-112.csv", BASE_DIR + model + "/Task4_AM_Qwen2.5-VL-7B-Instruct_1-112.csv")
#     # shutil.copy(BASE_DIR + "vlm/" + model + "/Task4_HT_Qwen2.5-VL-7B-Instruct_1-129.csv", BASE_DIR + model + "/Task4_HT_Qwen2.5-VL-7B-Instruct_1-129.csv")
#     # task_name = 'Task4L_5'
#     # subtask = 'Task4L'
#     # mergecsv(task_name, model,subtask)
#     # subtask = 'Task5'
#     # mergecsv(task_name, model,subtask)


#     # model = 'Qwen2.5-VL-32B-Instruct'
#     # task_name = 'Task1'
#     # mergecsv(task_name, model)      
#     # task_name = 'Task3_6'
#     # subtask = 'Task3'
#     # mergecsv(task_name, model,subtask)
#     # subtask = 'Task6'
#     # mergecsv(task_name, model,subtask)
#     # shutil.copy(BASE_DIR + "vlm/" + model + "/Task4_AM_Qwen2.5-VL-7B-Instruct_1-112.csv", BASE_DIR + model + "/Task4_AM_Qwen2.5-VL-7B-Instruct_1-112.csv")
#     # shutil.copy(BASE_DIR + "vlm/" + model + "/Task4_HT_Qwen2.5-VL-7B-Instruct_1-129.csv", BASE_DIR + model + "/Task4_HT_Qwen2.5-VL-7B-Instruct_1-129.csv")
#     # task_name = 'Task4L_5'
#     # subtask = 'Task4L'
#     # mergecsv(task_name, model,subtask)
#     # subtask = 'Task5'
#     # mergecsv(task_name, model,subtask)