import os
import re
import json
import asyncio
import aiohttp
from typing import List, Tuple
import pandas as pd

# ===================== Configuration =====================
# Read DashScope API key from env var; avoid hardcoding secrets.
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "YOUR API KEY"
# API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")  # set: export DASHSCOPE_API_KEY=xxx
MODEL = "qwen-plus"

# Project base (prefer env); replace <BASE_DIR> with your path if not using env.
BASE_DIR = os.environ.get("PROJECT_BASE_DIR", "<BASE_DIR>")

# IO patterns (must contain {model}); adjust to your actual file names.
# Example:
#   Input : <BASE_DIR>/Task6_segment2video/Task6_Qwen2.5-VL-72B-Instruct_all_merged.csv
#   Output: <BASE_DIR>/Task6_segment2video/Task6_Qwen2.5-VL-72B-Instruct_all_merged_llmmerge.csv
CSV_IN_PATTERN  = os.path.join(BASE_DIR, "Task6_segment2video", "Task6_{model}_all_merged.csv")
CSV_OUT_PATTERN = os.path.join(BASE_DIR, "Task6_segment2video", "Task6_{model}_all_merged_llmmerge.csv")

# Models to run (AF3 removed)
MODELS = [
    "InternVL3_5-8B",
    "InternVL3_5-38B",
    "Qwen2.5-VL-7B-Instruct",
    "Qwen2.5-VL-32B-Instruct",
    "Qwen2.5-VL-72B-Instruct",
    "Lingshu-32B",
]

# Processing settings
COL_PATTERN  = r"^report$"   # only merge the "report" column
TEMPERATURE  = 0
TOP_P        = 0.9
MAX_TOKENS   = 1024
TIMEOUT_SECS = 60.0
CONCURRENCY  = 10            # concurrency per model
RETRIES      = 3
BACKOFF_SECS = 1.5
# =========================================================

# ===================== Prompt Template ===================
PROMPT_TEMPLATE = (
    "The following is a list of report segments from a single observation. "
    "The segments are formatted as a bulleted list below: "
    "{segments_list} "
    "Merge them into a single consolidated report that is de-duplicated but information-preserving. "
    "Remove any duplicate or redundant sentences and merge statements with the same meaning. "
    'DO NOT include any introductory phrases or sentences like "Here is a consolidated report:". '
    "Start your response directly with the merged content."
)
# =========================================================


# --------------------- Helpers ---------------------------
def split_segments(cell_text: str) -> List[str]:
    """Split a cell by '||' and clean pieces."""
    if cell_text is None:
        return []
    parts = re.split(r"\s*\|\|\s*", str(cell_text).strip())
    return [p.strip() for p in parts if p and p.strip().lower() != "nan"]

def build_user_prompt(segments: List[str]) -> str:
    """Build user prompt with bullet list of segments."""
    segments_list_str = "\n- " + "\n- ".join(segments)
    return PROMPT_TEMPLATE.format(segments_list=segments_list_str)

async def call_qwen_async(session: aiohttp.ClientSession, user_prompt: str) -> str | None:
    """
    Call Qwen Chat Completions with retries.
    Return merged text on success; None after final failure.
    """
    url = f"{API_BASE.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": user_prompt}],
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
                text = (payload["choices"][0]["message"]["content"] or "").strip()
                # Trim accidental quotes/backticks and collapse whitespaces
                text = re.sub(r'^[\"“”`]+|[\"“”`]+$', "", text).strip()
                text = re.sub(r"\s+", " ", text)
                return text
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
            if attempt < RETRIES - 1:
                print(f"[WARN] Retry {attempt+1}/{RETRIES} in {BACKOFF_SECS}s. Error: {e}")
                await asyncio.sleep(BACKOFF_SECS)
            else:
                print(f"[ERROR] All {RETRIES} attempts failed. Error: {e}")
                return None

async def process_cell_async(
    task_index: int,
    total_tasks: int,
    row_index: int,
    col: str,
    row: pd.Series,
    session: aiohttp.ClientSession,
) -> tuple[int, str, str]:
    """
    Process one cell: build prompt, call API, return (row_idx, col, merged_text).
    On failure, fall back to original content.
    """
    original_content = row.get(col, "")
    if original_content is None or not str(original_content).strip():
        print(f"[{task_index+1}/{total_tasks}] Skip empty: Row {row_index}, Col '{col}'")
        return (row_index, col, "")

    segments = split_segments(str(original_content))
    if len(segments) <= 1:
        print(f"[{task_index+1}/{total_tasks}] Single segment -> keep: Row {row_index}, Col '{col}'")
        return (row_index, col, segments[0] if segments else "")

    user_prompt = build_user_prompt(segments)
    merged_text = await call_qwen_async(session, user_prompt)

    if merged_text is None:
        merged_text = original_content
        print(f"[INFO] API failed -> write original. ({task_index+1}/{total_tasks}) | Col='{col}'")
    else:
        print(f"[{task_index+1}/{total_tasks}] Done: Row {row_index}, Col '{col}'")

    return (row_index, col, merged_text)
# ---------------------------------------------------------


# ------------------ Per-Model Pipeline -------------------
async def process_one_model_async(model_name: str) -> None:
    """Run the whole merging pipeline for a single model."""
    csv_in  = CSV_IN_PATTERN.format(model=model_name)
    csv_out = CSV_OUT_PATTERN.format(model=model_name)
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    if not API_KEY:
        raise SystemExit("Please set DASHSCOPE_API_KEY in your environment.")

    # Read input with robust encoding fallback
    try:
        df = pd.read_csv(csv_in, encoding="utf-8")
    except UnicodeDecodeError:
        print(f"[WARN] {model_name}: UTF-8 failed. Trying GBK...")
        try:
            df = pd.read_csv(csv_in, encoding="gbk")
        except Exception as e:
            raise SystemExit(f"[FATAL] {model_name}: Failed to read CSV (utf-8 & gbk). Error: {e}")
    except FileNotFoundError:
        print(f"[WARN] Missing input for {model_name}: {csv_in}")
        return

    df_out = df.copy()

    # Select target columns by regex (only "report")
    rx = re.compile(COL_PATTERN)
    target_cols = [c for c in df.columns if rx.search(c)]
    if not target_cols:
        print(f"[INFO] {model_name}: No columns match pattern: {COL_PATTERN}")
        df_out.to_csv(csv_out, index=False, encoding="utf-8")
        print(f"[Done] {model_name} -> {csv_out} (no changes)")
        return

    # Build tasks for non-empty cells
    tasks_meta: List[Tuple[int, str, pd.Series]] = []
    for col in target_cols:
        for i in range(len(df)):
            raw_val = df.at[i, col]
            if raw_val and str(raw_val).strip():
                tasks_meta.append((i, col, df.iloc[i]))

    total_tasks = len(tasks_meta)
    if total_tasks == 0:
        print(f"[Info] {model_name}: No cells to process.")
        df_out.to_csv(csv_out, index=False, encoding="utf-8")
        print(f"[Done] {model_name} -> {csv_out} (no changes)")
        return

    print(f"[Info] {model_name}: Total cells to process = {total_tasks}")

    sem = asyncio.Semaphore(CONCURRENCY)

    async def limited_process(task_index, row_index, col, row, session):
        async with sem:
            return await process_cell_async(task_index, total_tasks, row_index, col, row, session)

    async with aiohttp.ClientSession() as session:
        all_tasks = [
            limited_process(i, row_index, col, row, session)
            for i, (row_index, col, row) in enumerate(tasks_meta)
        ]
        results = await asyncio.gather(*all_tasks)

    # Write results back to DataFrame
    for row_index, col, merged_text in results:
        df_out.at[row_index, col] = merged_text

    # Save CSV
    try:
        df_out.to_csv(csv_out, index=False, encoding="utf-8")
    except Exception as e:
        print(f"[ERROR] {model_name}: Failed to write UTF-8. Error: {e}. Falling back to default encoding.")
        df_out.to_csv(csv_out, index=False)

    print(f"[Done] {model_name} -> {csv_out}")
# ---------------------------------------------------------


# ------------------ Main Orchestrator --------------------
async def main_async():
    """Sequentially process all models (each model runs its own async batch)."""
    for m in MODELS:
        print(f"\n===== Processing model: {m} =====")
        await process_one_model_async(m)

if __name__ == "__main__":
    asyncio.run(main_async())
# ---------------------------------------------------------
