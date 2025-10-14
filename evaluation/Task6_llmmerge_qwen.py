import os
import re
import json
import asyncio
import aiohttp
from typing import List

import pandas as pd

# ========== CONFIGURATION - EDIT THESE VALUES ==========
# Qwen API settings
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "your API key"  # <-- Put your DashScope API key here
MODEL = "qwen-plus"

# File paths
CSV_IN = "/Users/pxy_amber/Downloads/Task6_segment2video/Task6_Qwen2.5-VL-72B-Instruct_all_merged.csv"
CSV_OUT = "/Users/pxy_amber/Downloads/Task6_segment2video/Task6_Qwen2.5-VL-72B-Instruct_all_merged_llmmerge.csv"

# Processing settings
COL_PATTERN = r"^report$"  # Regex pattern to match only the "report" column
TEMPERATURE = 0
TOP_P = 0.9
MAX_TOKENS = 1024
TIMEOUT_SECS = 60.0
CONCURRENCY = 10  # Number of concurrent API requests. Adjust based on your API limits.
RETRIES = 3  # Number of times to retry a failed API call
BACKOFF_SECS = 1.5  # Time in seconds to wait before retrying
# =======================================================

# A generalized prompt for merging any report.
PROMPT_TEMPLATE = (
    'The following is a list of report segments from a single observation. '
    'The segments are formatted as a bulleted list below: '
    '{segments_list} '
    'Merge them into a single consolidated report that is de-duplicated but information-preserving. '
    'Remove any duplicate or redundant sentences and merge statements with the same meaning. '
    'DO NOT include any introductory phrases or sentences like "Here is a consolidated report:". '
    'Start your response directly with the merged content.'
)


def split_segments(cell_text: str) -> List[str]:
    """Splits a string by '||' and cleans up the segments."""
    if cell_text is None:
        return []
    parts = re.split(r'\s*\|\|\s*', str(cell_text).strip())
    return [p.strip() for p in parts if p and p.strip().lower() != "nan"]


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
                text = re.sub(r'^[\"“”`]+|[\"“”`]+$', '', text).strip()
                text = re.sub(r'\s+', ' ', text)
                return text  # Return on success
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
            if attempt < RETRIES - 1:
                print(f"[WARN] Attempt {attempt + 1}/{RETRIES} failed. Retrying in {BACKOFF_SECS}s. Error: {e}")
                await asyncio.sleep(BACKOFF_SECS)
            else:
                print(f"[ERROR] All {RETRIES} attempts failed. Error: {e}")
                return None  # Return None to indicate final failure


async def process_cell_async(task_index: int, total_tasks: int, row_index: int, col: str, row: pd.Series, session):
    """
    Asynchronously processes a single cell: builds prompt, calls API, and returns result.
    If API calls fail after all retries, returns the original content.
    Includes progress logging.
    """
    original_content = row.get(col, "")
    if original_content is None or not str(original_content).strip():
        print(f"[{task_index + 1}/{total_tasks}] Skipping empty cell: Row {row_index}, Col '{col}'")
        return (row_index, col, "")

    segments = split_segments(str(original_content))
    if len(segments) <= 1:
        print(f"[{task_index + 1}/{total_tasks}] Skipping single-segment cell: Row {row_index}, Col '{col}'")
        return (row_index, col, segments[0] if segments else "")

    user_prompt = build_user_prompt(segments)

    merged_text = await call_qwen_async(session, user_prompt)

    if merged_text is None:
        merged_text = original_content
        print(f"[INFO] All API attempts for '{col}' failed. Writing original content. ({task_index + 1}/{total_tasks})")
    else:
        print(f"[{task_index + 1}/{total_tasks}] Task completed: Row {row_index}, Col '{col}'")

    return (row_index, col, merged_text)


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
            raise SystemExit(f"Failed to read CSV with both UTF-8 and GBK. Please check file encoding. Error: {e}")

    df_out = df.copy()

    rx = re.compile(COL_PATTERN)
    target_cols = [c for c in df.columns if rx.search(c)]
    if not target_cols:
        raise SystemExit(f"No columns match pattern: {COL_PATTERN}")

    tasks = []
    for col in target_cols:
        for i in range(len(df)):
            raw_val = df.at[i, col]
            if raw_val and str(raw_val).strip():
                tasks.append((i, col, df.iloc[i]))

    total_tasks = len(tasks)
    if total_tasks == 0:
        print("[Info] No cells to process.")
        return

    print(f"[Info] Total cells to process: {total_tasks}")

    sem = asyncio.Semaphore(CONCURRENCY)

    async def limited_process(task_index, row_index, col, row, session):
        async with sem:
            return await process_cell_async(task_index, total_tasks, row_index, col, row, session)

    async with aiohttp.ClientSession() as session:
        all_tasks = [limited_process(i, row_index, col, row, session) for i, (row_index, col, row) in enumerate(tasks)]

        results = await asyncio.gather(*all_tasks)

    for row_index, col, merged_text in results:
        df_out.at[row_index, col] = merged_text

    try:
        df_out.to_csv(CSV_OUT, index=False, encoding='utf-8')
    except Exception as e:
        print(
            f"[ERROR] Failed to write CSV file with UTF-8 encoding. Please check file path and permissions. Error: {e}")
        df_out.to_csv(CSV_OUT, index=False)

    print(f"[Done] Wrote: {CSV_OUT}")


if __name__ == "__main__":
    asyncio.run(main_async())
