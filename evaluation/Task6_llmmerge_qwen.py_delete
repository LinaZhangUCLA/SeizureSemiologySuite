#change input csv format

import os
import re
import json
import asyncio
import aiohttp
from typing import List, Dict
from collections import defaultdict

import pandas as pd

# ========== CONFIGURATION - EDIT THESE VALUES ==========
# Qwen API settings
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = 'sk-03164559b7d548da873c5d7a934a9059'  # <-- Put your DashScope API key here
MODEL = "qwen-plus"

# File paths
CSV_IN = "result/vlm_inference/Qwen2.5-VL-7B-Instruct/Task6_Qwen2.5-VL-7B-Instruct_1-2.csv"
CSV_OUT = "result/vlm_inference/Qwen2.5-VL-7B-Instruct/Task6_Qwen2.5-VL-72B-Instruct_all_merged_llmmerge.csv"

# Processing settings
TEMPERATURE = 0
TOP_P = 0.9
MAX_TOKENS = 1024
TIMEOUT_SECS = 60.0
CONCURRENCY = 10  # Number of concurrent API requests
RETRIES = 3  # Number of times to retry a failed API call
BACKOFF_SECS = 1.5  # Time in seconds to wait before retrying
# =======================================================

# Prompt for merging reports
PROMPT_TEMPLATE = (
    'The following is a list of report segments from a single observation. '
    'The segments are formatted as a bulleted list below: '
    '{segments_list} '
    'Merge them into a single consolidated report that is de-duplicated but information-preserving. '
    'Remove any duplicate or redundant sentences and merge statements with the same meaning. '
    'DO NOT include any introductory phrases or sentences like "Here is a consolidated report:". '
    'Start your response directly with the merged content.'
)


def extract_video_base_name(video_name: str) -> str:
    """
    Extract the base video name without segment suffix.
    Example: 'A0002@5-13-2021@UA6693LK@sz_v1_1_segment_0.mp4' -> 'A0002@5-13-2021@UA6693LK@sz_v1_1'
    """
    # Remove .mp4 extension
    name = video_name.replace('.mp4', '')
    # Remove _segment_X suffix
    base_name = re.sub(r'_segment_\d+$', '', name)
    return base_name


def group_by_video(df: pd.DataFrame) -> Dict[str, List[tuple]]:
    """
    Group rows by base video name.
    Returns a dict: {base_video_name: [(index, video_name, report), ...]}
    """
    groups = defaultdict(list)
    
    for idx, row in df.iterrows():
        video_name = row.get('video_name', '')
        report = row.get('report', '')
        
        if not video_name or not report or str(report).strip() == '':
            continue
            
        base_name = extract_video_base_name(str(video_name))
        groups[base_name].append((idx, video_name, str(report).strip()))
    
    return groups


def build_user_prompt(segments: List[str]) -> str:
    """Constructs the full prompt for the Qwen API using a bulleted list."""
    segments_list_str = '\n- ' + '\n- '.join(segments)
    return PROMPT_TEMPLATE.format(segments_list=segments_list_str)


async def call_qwen_async(session, user_prompt: str):
    """
    Asynchronously calls the Qwen API with a retry mechanism.
    Returns the merged text on success, or None on failure after all retries.
    """
    url = f"{API_BASE.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": user_prompt},
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "n": 1,
    }
    
    for attempt in range(RETRIES):
        try:
            async with session.post(url, headers=headers, json=body, timeout=TIMEOUT_SECS) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {text}")

                payload = await resp.json()
                text = payload["choices"][0]["message"]["content"]
                text = (text or "").strip()
                text = re.sub(r'^[\"""`]+|[\"""`]+$', '', text).strip()
                text = re.sub(r'\s+', ' ', text)
                return text
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
            if attempt < RETRIES - 1:
                print(f"[WARN] Attempt {attempt + 1}/{RETRIES} failed. Retrying in {BACKOFF_SECS}s. Error: {e}")
                await asyncio.sleep(BACKOFF_SECS)
            else:
                print(f"[ERROR] All {RETRIES} attempts failed. Error: {e}")
                return None


async def process_group_async(task_index: int, total_tasks: int, base_name: str, 
                               group_data: List[tuple], session):
    """
    Process a group of segments belonging to the same video.
    Returns: (base_name, merged_text, [(index, original_report), ...])
    """
    if len(group_data) == 1:
        idx, video_name, report = group_data[0]
        print(f"[{task_index + 1}/{total_tasks}] Skipping single-segment video: {base_name}")
        return (base_name, report, [(idx, report)])
    
    # Extract reports for merging
    reports = [report for _, _, report in group_data]
    user_prompt = build_user_prompt(reports)
    
    merged_text = await call_qwen_async(session, user_prompt)
    
    if merged_text is None:
        # If API fails, concatenate with ||
        merged_text = " || ".join(reports)
        print(f"[INFO] API failed for '{base_name}'. Using concatenated text. ({task_index + 1}/{total_tasks})")
    else:
        print(f"[{task_index + 1}/{total_tasks}] Merged {len(reports)} segments for: {base_name}")
    
    # Return indices and their new merged text
    indices_and_originals = [(idx, report) for idx, _, report in group_data]
    return (base_name, merged_text, indices_and_originals)


async def main_async():
    """Main asynchronous function to orchestrate the entire process."""
    if not API_KEY or API_KEY == "YOUR_DASHSCOPE_API_KEY_HERE":
        raise SystemExit("Please edit this file and set API_KEY to your DashScope key.")

    try:
        df = pd.read_csv(CSV_IN, encoding='utf-8')
    except UnicodeDecodeError:
        print("[WARN] UTF-8 decoding failed. Trying alternative encodings...")
        try:
            df = pd.read_csv(CSV_IN, encoding='gbk')
        except Exception as e:
            raise SystemExit(f"Failed to read CSV. Error: {e}")

    # Check required columns
    if 'video_name' not in df.columns or 'report' not in df.columns:
        raise SystemExit("CSV must contain 'video_name' and 'report' columns")

    df_out = df.copy()
    
    # Group by base video name
    video_groups = group_by_video(df)
    
    total_tasks = len(video_groups)
    if total_tasks == 0:
        print("[Info] No videos to process.")
        return

    print(f"[Info] Total video groups to process: {total_tasks}")
    
    sem = asyncio.Semaphore(CONCURRENCY)

    async def limited_process(task_index, base_name, group_data, session):
        async with sem:
            return await process_group_async(task_index, total_tasks, base_name, group_data, session)

    async with aiohttp.ClientSession() as session:
        all_tasks = [
            limited_process(i, base_name, group_data, session) 
            for i, (base_name, group_data) in enumerate(video_groups.items())
        ]
        results = await asyncio.gather(*all_tasks)

    # Update the dataframe with merged results
    for base_name, merged_text, indices_and_originals in results:
        for idx, _ in indices_and_originals:
            df_out.at[idx, 'report'] = merged_text

    try:
        df_out.to_csv(CSV_OUT, index=False, encoding='utf-8')
    except Exception as e:
        print(f"[ERROR] Failed to write CSV file. Error: {e}")
        df_out.to_csv(CSV_OUT, index=False)

    print(f"[Done] Wrote: {CSV_OUT}")


if __name__ == "__main__":
    asyncio.run(main_async())



#previous
# import os
# import re
# import json
# import asyncio
# import aiohttp
# from typing import List

# import pandas as pd

# # ========== CONFIGURATION - EDIT THESE VALUES ==========
# # Qwen API settings
# API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# API_KEY = 'sk-03164559b7d548da873c5d7a934a9059'  # <-- Put your DashScope API key here
# MODEL = "qwen-plus"

# # File paths
# CSV_IN = "/Users/pxy_amber/Downloads/Task6_segment2video/Task6_Qwen2.5-VL-72B-Instruct_all_merged.csv"
# CSV_OUT = "/Users/pxy_amber/Downloads/Task6_segment2video/Task6_Qwen2.5-VL-72B-Instruct_all_merged_llmmerge.csv"

# # Processing settings
# COL_PATTERN = r"^report$"  # Regex pattern to match only the "report" column
# TEMPERATURE = 0
# TOP_P = 0.9
# MAX_TOKENS = 1024
# TIMEOUT_SECS = 60.0
# CONCURRENCY = 10  # Number of concurrent API requests. Adjust based on your API limits.
# RETRIES = 3  # Number of times to retry a failed API call
# BACKOFF_SECS = 1.5  # Time in seconds to wait before retrying
# # =======================================================

# # A generalized prompt for merging any report.
# PROMPT_TEMPLATE = (
#     'The following is a list of report segments from a single observation. '
#     'The segments are formatted as a bulleted list below: '
#     '{segments_list} '
#     'Merge them into a single consolidated report that is de-duplicated but information-preserving. '
#     'Remove any duplicate or redundant sentences and merge statements with the same meaning. '
#     'DO NOT include any introductory phrases or sentences like "Here is a consolidated report:". '
#     'Start your response directly with the merged content.'
# )


# def split_segments(cell_text: str) -> List[str]:
#     """Splits a string by '||' and cleans up the segments."""
#     if cell_text is None:
#         return []
#     parts = re.split(r'\s*\|\|\s*', str(cell_text).strip())
#     return [p.strip() for p in parts if p and p.strip().lower() != "nan"]


# def build_user_prompt(segments: List[str]) -> str:
#     """Constructs the full prompt for the Qwen API using a bulleted list."""
#     segments_list_str = '\n- ' + '\n- '.join(segments)

#     return PROMPT_TEMPLATE.format(segments_list=segments_list_str)


# async def call_qwen_async(session, user_prompt: str):
#     """
#     Asynchronously calls the Qwen API with a retry mechanism.
#     Returns the merged text on success, or None on failure after all retries.
#     """
#     url = f"{API_BASE.rstrip('/')}/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {API_KEY}",
#         "Content-Type": "application/json"
#     }
#     body = {
#         "model": MODEL,
#         "messages": [
#             {"role": "user", "content": user_prompt},
#         ],
#         "temperature": TEMPERATURE,
#         "top_p": TOP_P,
#         "max_tokens": MAX_TOKENS,
#         "n": 1,
#     }
#     for attempt in range(RETRIES):
#         try:
#             async with session.post(url, headers=headers, json=body, timeout=TIMEOUT_SECS) as resp:
#                 if resp.status != 200:
#                     text = await resp.text()
#                     raise RuntimeError(f"HTTP {resp.status}: {text}")

#                 payload = await resp.json()
#                 text = payload["choices"][0]["message"]["content"]
#                 text = (text or "").strip()
#                 text = re.sub(r'^[\"“”`]+|[\"“”`]+$', '', text).strip()
#                 text = re.sub(r'\s+', ' ', text)
#                 return text  # Return on success
#         except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
#             if attempt < RETRIES - 1:
#                 print(f"[WARN] Attempt {attempt + 1}/{RETRIES} failed. Retrying in {BACKOFF_SECS}s. Error: {e}")
#                 await asyncio.sleep(BACKOFF_SECS)
#             else:
#                 print(f"[ERROR] All {RETRIES} attempts failed. Error: {e}")
#                 return None  # Return None to indicate final failure


# async def process_cell_async(task_index: int, total_tasks: int, row_index: int, col: str, row: pd.Series, session):
#     """
#     Asynchronously processes a single cell: builds prompt, calls API, and returns result.
#     If API calls fail after all retries, returns the original content.
#     Includes progress logging.
#     """
#     original_content = row.get(col, "")
#     if original_content is None or not str(original_content).strip():
#         print(f"[{task_index + 1}/{total_tasks}] Skipping empty cell: Row {row_index}, Col '{col}'")
#         return (row_index, col, "")

#     segments = split_segments(str(original_content))
#     if len(segments) <= 1:
#         print(f"[{task_index + 1}/{total_tasks}] Skipping single-segment cell: Row {row_index}, Col '{col}'")
#         return (row_index, col, segments[0] if segments else "")

#     user_prompt = build_user_prompt(segments)

#     merged_text = await call_qwen_async(session, user_prompt)

#     if merged_text is None:
#         merged_text = original_content
#         print(f"[INFO] All API attempts for '{col}' failed. Writing original content. ({task_index + 1}/{total_tasks})")
#     else:
#         print(f"[{task_index + 1}/{total_tasks}] Task completed: Row {row_index}, Col '{col}'")

#     return (row_index, col, merged_text)


# async def main_async():
#     """Main asynchronous function to orchestrate the entire process."""
#     if not API_KEY or API_KEY == "YOUR_DASHSCOPE_API_KEY_HERE":
#         raise SystemExit("Please edit this file and set API_KEY to your DashScope key.")

#     try:
#         df = pd.read_csv(CSV_IN, encoding='utf-8')
#     except UnicodeDecodeError:
#         print("[WARN] UTF-8 decoding failed. Trying alternative encodings...")
#         try:
#             df = pd.read_csv(CSV_IN, encoding='gbk')
#         except Exception as e:
#             raise SystemExit(f"Failed to read CSV with both UTF-8 and GBK. Please check file encoding. Error: {e}")

#     df_out = df.copy()

#     rx = re.compile(COL_PATTERN)
#     target_cols = [c for c in df.columns if rx.search(c)]
#     if not target_cols:
#         raise SystemExit(f"No columns match pattern: {COL_PATTERN}")

#     tasks = []
#     for col in target_cols:
#         for i in range(len(df)):
#             raw_val = df.at[i, col]
#             if raw_val and str(raw_val).strip():
#                 tasks.append((i, col, df.iloc[i]))

#     total_tasks = len(tasks)
#     if total_tasks == 0:
#         print("[Info] No cells to process.")
#         return

#     print(f"[Info] Total cells to process: {total_tasks}")

#     sem = asyncio.Semaphore(CONCURRENCY)

#     async def limited_process(task_index, row_index, col, row, session):
#         async with sem:
#             return await process_cell_async(task_index, total_tasks, row_index, col, row, session)

#     async with aiohttp.ClientSession() as session:
#         all_tasks = [limited_process(i, row_index, col, row, session) for i, (row_index, col, row) in enumerate(tasks)]

#         results = await asyncio.gather(*all_tasks)

#     for row_index, col, merged_text in results:
#         df_out.at[row_index, col] = merged_text

#     try:
#         df_out.to_csv(CSV_OUT, index=False, encoding='utf-8')
#     except Exception as e:
#         print(
#             f"[ERROR] Failed to write CSV file with UTF-8 encoding. Please check file path and permissions. Error: {e}")
#         df_out.to_csv(CSV_OUT, index=False)

#     print(f"[Done] Wrote: {CSV_OUT}")


# if __name__ == "__main__":
#     asyncio.run(main_async())
