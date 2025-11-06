import os
import re
import glob
import pandas as pd
import csv
import shutil


# === Set your input folder and output path ===
# Get the script directory and project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BASE_DIR = PROJECT_ROOT
DEDUP_ON_COLUMN = 'file_name' 
origin_dir = os.path.join(BASE_DIR, "result",  "vlm_inference_test","seizure_omni_sft","raw")
output_dir = os.path.join(BASE_DIR, "result", "vlm_inference_test",)


# ============================================

# def read_csv_file(input_file: str) -> pd.DataFrame:
#     corrected_data = []
#     with open(input_file, 'r', newline='', encoding='utf-8') as infile:
#         reader = csv.reader(infile)     
#         header = next(reader)      
#         for row in reader:
#             if len(row) > 2:
#                 first_two_cols = row[:1]
#                 merged_col = ','.join(row[1:])
#                 corrected_row = first_two_cols + [merged_col]
#                 corrected_data.append(corrected_row)
#             else:
#                 corrected_data.append(row)
#     df = pd.DataFrame(corrected_data, columns=header)
#     return df



def _clean_second_col(text: str) -> str:
    # 保留开头和结尾的引号，把中间的引号都去掉
    if text.startswith('"') and text.endswith('"'):
        inner = text[1:-1].replace('"', '')   # 中间所有 " 去掉
        return f'"{inner}"'
    else:
        # 不带外层引号的情况，就直接把里面的 " 去掉
        return text.replace('"', '')

def read_csv_file(input_file: str) -> pd.DataFrame:
    corrected_data = []
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)

        for row in reader:
            if len(row) > 2:
                # 只有第一列是独立的，后面都合并成第二列
                first_col = row[0]
                merged_col = ' '.join(row[1:])
                merged_col = _clean_second_col(merged_col)
                corrected_data.append([first_col, merged_col])
            else:
                # 正常两列的情况，也要清洗第二列
                if len(row) == 2:
                    row[1] = _clean_second_col(row[1])
                corrected_data.append(row)

    df = pd.DataFrame(corrected_data, columns=header)
    return df

import re

def keep_es_or_nes(text: str) -> str:
    """
    只保留开头的 ES 或 NES，后面内容都去掉。
    """
    if text is None:
        return text
    # 去掉前后空格
    text = text.strip()
    m = re.match(r'^(ES|NES)\b', text, flags=re.IGNORECASE)
    if m:
        # 返回和原来大小写一致的那段
        return m.group(1)
    else:
        # 不是 ES/NES 的就原样返回，也可以返回空串，看你想要什么
        return text
    


def mergecsv(task_name: str, model: str,subtask:str=None):
    INPUT_DIR = origin_dir #+ model
    # Match the target CSVs
    print(f"{task_name}_{model}_*.csv")
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
        if task_name=='Task6':
            df = read_csv_file(fp)
        else:
            df = pd.read_csv(fp, dtype=str)          # use string dtype to avoid type mismatches
            if task_name=='Task7':
                df['prediction_with_report'] = df['prediction_with_report'].apply(keep_es_or_nes) 
                df['prediction_without_report'] = df['prediction_without_report'].apply(keep_es_or_nes)
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
    if subtask=='Task4':    
        merged = merged[["video_name", "feature","timestamp"]] 
        merged = merged.rename(columns={"video_name":"file_name"})
    if subtask=='Task5':    
        merged = merged[["video_name", "event_sequence"]] 
        merged = merged.rename(columns={"video_name": "file_name"})
    if subtask=='Task7':    
        merged = merged[["video_name", "prediction_with_report", "prediction_without_report"]]
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
    OUTPUT_CSV = os.path.join(output_dir, model, f"{subtask}_{model}_all.csv")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    merged.to_csv(OUTPUT_CSV, index=False)
    print("Merge complete. Saved to:", OUTPUT_CSV)
    print("Total rows:", len(merged))
    
    # error_log = os.path.join(output_dir, 'row_error_file_stat.log')
    # if not os.path.exists(error_log):
    #     with open(error_log, 'w') as f:
    #         f.write("Row count errors:\n")
    # print(error_log)
    # if subtask == 'Task1':
    #    if len(merged) != 2314:
    #        #write to row_error.log
    #           with open(error_log, "a") as f:
    #               f.write(f"{OUTPUT_CSV} has {len(merged)} rows, expected 2314\n")
    #               print(f"!!!!!!!!!!!!{OUTPUT_CSV} has {len(merged)} rows, expected 2314\n")
    # if subtask == 'Task3' or subtask == 'Task6':
    #    if len(merged) != 2316:
    #        #write to row_error.log
    #           with open(error_log, "a") as f:
    #               f.write(f"{OUTPUT_CSV} has {len(merged)} rows, expected 2316\n")
    #               print(f"!!!!!!!!!!!!{OUTPUT_CSV} has {len(merged)} rows, expected 2316\n")
    # if subtask == 'Task5' or subtask == 'Task4L':
    #    if len(merged) != 2413:
    #        #write to row_error.log
    #           with open(error_log, "a") as f:
    #               f.write(f"{OUTPUT_CSV} has {len(merged)} rows, expected 2413\n")
    #               print(f"!!!!!!!!!!!!{OUTPUT_CSV} has {len(merged)} rows, expected 2413\n")
    # if subtask == 'Task7':
    #    if len(merged) != 438:
    #        #write to row_error.log
    #           with open(error_log, "a") as f:
    #               f.write(f"{OUTPUT_CSV} has {len(merged)} rows, expected 438\n")
    #               print(f"!!!!!!!!!!!!{OUTPUT_CSV} has {len(merged)} rows, expected 438\n")    




if __name__ == "__main__":

    # for model in ['InternVL3_5-8B','InternVL3_5-38B','Qwen2.5-VL-7B-Instruct','Qwen2.5-VL-32B-Instruct','Qwen2.5-VL-72B-Instruct', 'Lingshu-32B','Qwen2.5-Omni-7B']:
        
    #     output_modle_dir = os.path.join(output_dir, model)
    #     os.makedirs(output_modle_dir, exist_ok=True)

    #     task_name = 'Task3_6'
    #     subtask = 'Task3'
    #     mergecsv(task_name, model,subtask)
    #     subtask = 'Task6'
    #     mergecsv(task_name, model,subtask)
 
    #     task_name = 'Task4L_5'
    #     subtask = 'Task4L'
    #     mergecsv(task_name, model,subtask)
    #     subtask = 'Task5'
    #     mergecsv(task_name, model,subtask)
        
    #     # Task7 processing
    #     try:
    #         task_name = 'Task7'
    #         subtask = 'Task7'
    #         mergecsv(task_name, model, subtask)
    #     except FileNotFoundError as e:
    #         print(f"Skipping Task7 for {model}: {e}")

    # model = 'InternVL3_5-8B'
    # task_name = 'Task1'
    # mergecsv(task_name, model)      
    # task_name = 'Task3_6'
    # subtask = 'Task3'
    # mergecsv(task_name, model,subtask)
    # subtask = 'Task6'
    # mergecsv(task_name, model,subtask)
    # shutil.copy(BASE_DIR + "vlm/" + model + "/Task4_AM_Qwen2.5-VL-7B-Instruct_1-112.csv", BASE_DIR + model + "/Task4_AM_Qwen2.5-VL-7B-Instruct_1-112.csv")
    # shutil.copy(BASE_DIR + "vlm/" + model + "/Task4_HT_Qwen2.5-VL-7B-Instruct_1-129.csv", BASE_DIR + model + "/Task4_HT_Qwen2.5-VL-7B-Instruct_1-129.csv")
    # task_name = 'Task4L_5'
    # subtask = 'Task4L'
    # mergecsv(task_name, model,subtask)
    # subtask = 'Task5'
    # mergecsv(task_name, model,subtask)


    # model = 'Qwen2.5-VL-32B-Instruct'
    # task_name = 'Task1'
    # mergecsv(task_name, model)      
    # task_name = 'Task3_6'
    # subtask = 'Task3'
    # mergecsv(task_name, model,subtask)
    # subtask = 'Task6'
    # mergecsv(task_name, model,subtask)
    # shutil.copy(BASE_DIR + "vlm/" + model + "/Task4_AM_Qwen2.5-VL-7B-Instruct_1-112.csv", BASE_DIR + model + "/Task4_AM_Qwen2.5-VL-7B-Instruct_1-112.csv")
    # shutil.copy(BASE_DIR + "vlm/" + model + "/Task4_HT_Qwen2.5-VL-7B-Instruct_1-129.csv", BASE_DIR + model + "/Task4_HT_Qwen2.5-VL-7B-Instruct_1-129.csv")
    # task_name = 'Task4L_5'
    # subtask = 'Task4L'
    # mergecsv(task_name, model,subtask)
    # subtask = 'Task5'
    # mergecsv(task_name, model,subtask)

   
    ##task12
    #for model in ['Qwen3-VL-8B-Instruct','Qwen3-VL-32B-Instruct','Qwen3-Omni-30B-A3B-Instruct'] + ['InternVL3_5-8B','InternVL3_5-38B','Qwen2.5-VL-7B-Instruct','Qwen2.5-VL-32B-Instruct','Qwen2.5-VL-72B-Instruct', 'Lingshu-32B','Qwen2.5-Omni-7B']: 
    for model in ['seizure_omni_sft']: 
        if model in ["Lingshu-32B",'InternVL3_5-38B']:
            continue
        task_name = 'Task1'
        mergecsv(task_name, model,subtask=task_name) 


    # #task3
    # for model in ['Qwen3-VL-8B-Instruct','Qwen3-VL-32B-Instruct','Qwen3-Omni-30B-A3B-Instruct']:
    #     output_modle_dir = os.path.join(output_dir, model)
    #     os.makedirs(output_modle_dir, exist_ok=True)
    #     shutil.copy(os.path.join(origin_dir, f"Task3_AM_{model}_1-112.csv"), os.path.join(output_dir, model, f"Task3_AM_{model}.csv"))
    #     shutil.copy(os.path.join(origin_dir, f"Task3_HT_{model}_1-129.csv"), os.path.join(output_dir, model, f"Task3_HT_{model}.csv"))
    #     shutil.copy(os.path.join(origin_dir, f"Task3_L_{model}_1-112.csv"), os.path.join(output_dir, model, f"Task3_L_{model}.csv"))
    
    # for model in ['InternVL3_5-38B','Qwen2.5-VL-7B-Instruct','Qwen2.5-VL-32B-Instruct','Qwen2.5-VL-72B-Instruct', 'Lingshu-32B','Qwen2.5-Omni-7B']: 
    #     output_modle_dir = os.path.join(output_dir, model)

    #     input_path =  f"{output_dir}/{model}/Task3_bodypart_{model}_all.csv"   # 换成你的原文件路径
    #     df = pd.read_csv(input_path)
    #     # 2. 只保留 file_name 中包含 "_segment_0" 的行
    #     mask = df["file_name"].str.contains("_segment_0")
    #     df = df[mask].copy()
    #     # 3. 把 file_name 里的 "_segment_0" 去掉
    #     df["file_name"] = df["file_name"].str.replace("_segment_0", "", regex=False)

    #     if(len(df) != 438):
    #         print(model, "task3_l length: ", len(df))
    #     # 4. 保存新的 csv
    #     output_path = f"{output_modle_dir}/Task3_L_{model}.csv"
    #     df.to_csv(output_path, index=False)

    #     if os.path.exists(input_path):
    #        os.remove(input_path)

    #     shutil.move(os.path.join(output_modle_dir, f"Task3_AM_{model}_1-112.csv"), os.path.join(output_dir, model, f"Task3_AM_{model}.csv"))
    #     shutil.move(os.path.join(output_modle_dir, f"Task3_HT_{model}_1-129.csv"), os.path.join(output_dir, model, f"Task3_HT_{model}.csv"))

    # #task4     
    #for model in ['Qwen3-VL-8B-Instruct']:
    # for model in ['Qwen3-VL-8B-Instruct','Qwen3-VL-32B-Instruct','Qwen3-Omni-30B-A3B-Instruct'] + ['InternVL3_5-8B','InternVL3_5-38B','Qwen2.5-VL-7B-Instruct','Qwen2.5-VL-32B-Instruct','Qwen2.5-VL-72B-Instruct', 'Lingshu-32B','Qwen2.5-Omni-7B']: 
    #     if model in ["Lingshu-32B",'InternVL3_5-38B']:
    #         continue
    #     task_name = 'Task4'
    #     mergecsv(task_name, model,subtask=task_name)   

    
    # #task5

    #for model in ['Qwen3-VL-8B-Instruct']:
    # for model in ['Qwen3-VL-8B-Instruct','Qwen3-VL-32B-Instruct','Qwen3-Omni-30B-A3B-Instruct'] : 
    #     task_name = 'Task5'
    #     mergecsv(task_name, model,subtask=task_name)   

    ##task6
    #for model in ['Qwen3-VL-8B-Instruct','Qwen3-VL-32B-Instruct','Qwen3-Omni-30B-A3B-Instruct'] + ['InternVL3_5-8B','InternVL3_5-38B','Qwen2.5-VL-7B-Instruct','Qwen2.5-VL-32B-Instruct','Qwen2.5-VL-72B-Instruct', 'Lingshu-32B','Qwen2.5-Omni-7B']: 
    # for model in ['InternVL3_5-8B']:
        # if model in ["Lingshu-32B",'InternVL3_5-38B']:
        #     continue
        # task_name = 'Task6'
        # mergecsv(task_name, model,subtask=task_name) 


    ##task7
    # for model in ['Qwen3-VL-8B-Instruct','Qwen3-VL-32B-Instruct','Qwen3-Omni-30B-A3B-Instruct'] + ['InternVL3_5-8B','InternVL3_5-38B','Qwen2.5-VL-7B-Instruct','Qwen2.5-VL-32B-Instruct','Qwen2.5-VL-72B-Instruct', 'Lingshu-32B','Qwen2.5-Omni-7B']: 
    #     if model in ["Lingshu-32B",'InternVL3_5-38B']:
    #         continue
    #     task_name = 'Task7'
    #     mergecsv(task_name, model,subtask=task_name) 
