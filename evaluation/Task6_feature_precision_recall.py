import os
import re
import json
import asyncio
import aiohttp
from typing import List, Tuple
import pandas as pd

# ===================== CONFIG =====================
# DashScope (Qwen) API — use an env var in practice:
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "YOUR_API_KEY"  # or: os.environ.get("DASHSCOPE_API_KEY", "").strip()
MODEL = "qwen-plus"

# Project base (hidden path). Prefer environment variable; otherwise replace <BASE_DIR>.
BASE_DIR = os.environ.get("PROJECT_BASE_DIR", "<BASE_DIR>")

# I/O patterns (must contain {model}). Adjust to match your filenames.
# Example:
#   Input : <BASE_DIR>/task6_metrics/Task6_llmmerge/Task6_{model}_all_merged_llmmerge.csv
#   Output: <BASE_DIR>/task6_metrics/Task6_llmmerge/Task6_{model}_extracted_features_qwen.csv
CSV_IN_PATTERN  = os.path.join(BASE_DIR, "task6_metrics", "Task6_llmmerge", "Task6_{model}_all_merged_llmmerge.csv")
CSV_OUT_PATTERN = os.path.join(BASE_DIR, "task6_metrics", "Task6_llmmerge", "Task6_{model}_extracted_features_qwen.csv")

# Run these six models in sequence (AF3 excluded)
MODELS = [
    "InternVL3_5-8B",
    "InternVL3_5-38B",
    "Qwen2.5-VL-7B-Instruct",
    "Qwen2.5-VL-32B-Instruct",
    "Qwen2.5-VL-72B-Instruct",
    "Lingshu-32B",
]

# Concurrency & request settings
CONCURRENCY = 20
TIMEOUT_SECS = 60.0
RETRIES = 3
BACKOFF_SECS = 1.5
TEMPERATURE = 0
TOP_P = 0.9
MAX_TOKENS = 8   # very short output for YES/NO
# ===================================================

# Features to judge per report (column "report" is expected in input)
SEMIOLOGY_FEATURES = [
    "occur_during_sleep", "head_turning", "blank_stare", "close_eyes",
    "eye_blinking", "face_pulling", "face_twitching", "tonic", "clonic",
    "arm_straightening", "arm_flexion", "figure4", "oral_automatisms",
    "limb_automatisms", "asynchronous_movement", "pelvic_thrusting",
    "full_body_shaking", "arms_move_simultaneously", "verbal_responsiveness",
    "ictal_vocalization",
]

# ===================== Prompt =====================
PROMPT_TEMPLATE = (
    'You are an expert clinical neurologist. Determine if the following seizure report explicitly mentions '
    'or describes the feature: "{feature_readable}".\n\n'
    "[Clinical Report]\n"
    '\"\"\"{report}\"\"\"\n\n'
    "Answer with a single word ONLY: YES or NO."
)


# ===================== Helpers =====================
def build_prompt(report_text: str, feature_name: str) -> str:
    return PROMPT_TEMPLATE.format(
        feature_readable=feature_name.replace("_", " ").title(),
        report=(report_text or "").strip(),
    )

def parse_yes_no(content: str):
    """
    Use the first non-empty token:
      YES -> 1
      NO  -> 0
      else -> 'NA' (string)
    """
    if not content:
        return "NA"
    head = re.split(r"\s+", content.strip().upper())[0]
    if head == "YES":
        return 1
    if head == "NO":
        return 0
    return "NA"

async def call_qwen_yesno(session: aiohttp.ClientSession, user_prompt: str) -> Tuple[bool, str]:
    """
    Call Qwen chat completions with a small retry loop.
    Returns (ok, text). Falls back to removing unsupported sampling params.
    """
    url = f"{API_BASE.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    base = {
        "model": MODEL,
        "messages": [{"role": "user", "content": user_prompt}],
        "max_tokens": MAX_TOKENS,
        "n": 1,
    }
    bodies = [
        {**base, "temperature": TEMPERATURE, "top_p": TOP_P},
        {**base},  # fallback w/o sampling params
    ]

    for attempt in range(RETRIES):
        for variant_idx, body in enumerate(bodies):
            try:
                async with session.post(url, headers=headers, json=body, timeout=TIMEOUT_SECS) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        # if sampling params unsupported, try fallback body
                        if "Unsupported parameter" in text and variant_idx == 0:
                            continue
                        raise RuntimeError(f"HTTP {resp.status}: {text}")

                    payload = await resp.json()
                    text = (payload.get("choices", [{}])[0]
                                   .get("message", {})
                                   .get("content", "") or "").strip()
                    # strip quotes and collapse spaces
                    text = re.sub(r'^[\"“”`]+|[\"“”`]+$', "", text).strip()
                    text = re.sub(r"\s+", " ", text)
                    return True, text
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
                # if sampling unsupported, immediately try fallback body
                if "Unsupported parameter" in str(e) and variant_idx == 0:
                    continue
                if attempt < RETRIES - 1:
                    await asyncio.sleep(BACKOFF_SECS * (attempt + 1))
                else:
                    return False, f"[ERROR] {e}"

    return False, "[ERROR] Exhausted retries"

async def judge_one_feature_task(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    row_index: int,
    feature: str,
    report_text: str,
) -> Tuple[int, str, object]:
    """
    Judge a single (row, feature). Returns (row_index, feature, 1/0/'NA').
    """
    if not isinstance(report_text, str) or not report_text.strip():
        return row_index, feature, "NA"

    prompt = build_prompt(report_text, feature)
    async with sem:
        ok, text = await call_qwen_yesno(session, prompt)

    if not ok:
        return row_index, feature, "NA"

    return row_index, feature, parse_yes_no(text)


# ===================== Per-Model Runner =====================
async def run_one_model(model_name: str) -> None:
    """
    Read the model-specific CSV, run YES/NO judgments for all features, and write output CSV.
    Input CSV must contain columns: 'file_name', 'report'.
    """
    csv_in  = CSV_IN_PATTERN.format(model=model_name)
    csv_out = CSV_OUT_PATTERN.format(model=model_name)
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)

    if not API_KEY:
        raise SystemExit("DASHSCOPE_API_KEY / YOUR_API_KEY is empty. Please set it before running.")

    # Read input
    try:
        df = pd.read_csv(csv_in)
    except FileNotFoundError:
        print(f"[WARN] Missing input for {model_name}: {csv_in}")
        return

    # Basic schema check
    if "file_name" not in df.columns or "report" not in df.columns:
        print(f"[WARN] {model_name}: Input must contain columns 'file_name' and 'report'.")
        return

    # Prepare output (default to 'NA')
    df_out = pd.DataFrame({"file_name": df["file_name"]})
    for f in SEMIOLOGY_FEATURES:
        df_out[f] = "NA"

    # Build tasks
    tasks = []
    sem = asyncio.Semaphore(CONCURRENCY)
    async with aiohttp.ClientSession() as session:
        for i, row in df.iterrows():
            report_text = row.get("report", "")
            for feature in SEMIOLOGY_FEATURES:
                tasks.append(judge_one_feature_task(sem, session, i, feature, report_text))

        total = len(tasks)
        print(f"[Info] {model_name}: Total feature-judgement calls: {total} "
              f"({len(df)} reports × {len(SEMIOLOGY_FEATURES)} features)")

        # Execute in manageable chunks
        results: List[Tuple[int, str, object]] = []
        for start in range(0, total, CONCURRENCY * 5):
            chunk = tasks[start:start + CONCURRENCY * 5]
            results.extend(await asyncio.gather(*chunk))
            print(f"[Progress] {model_name}: {min(len(results), total)}/{total} done")

    # Fill results
    for row_index, feature, val in results:
        df_out.at[row_index, feature] = val  # 1 / 0 / 'NA'

    # Write output
    df_out.to_csv(csv_out, index=False)
    print(f"[Done] {model_name} -> {csv_out}")


# ===================== Orchestrator =====================
async def main_async():
    for m in MODELS:
        print(f"\n===== Processing model: {m} =====")
        await run_one_model(m)

if __name__ == "__main__":
    asyncio.run(main_async())
