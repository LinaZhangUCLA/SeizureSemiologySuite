import os
import re
import json
import asyncio
import aiohttp
from typing import List, Tuple

import pandas as pd

# ===================== USER CONFIG =====================
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "your api kay"
MODEL = "qwen-plus"
# API_KEY = os.environ.get("DASHSCOPE_API_KEY", "").strip()

INFERENCE_FILE = "/Users/pxy_amber/Downloads/task6_metrics/Task6_llmmerge/Task6_InternVL3_5-8B_all_merged_llmmerge.csv"
OUTPUT_FILE = "/Users/pxy_amber/Downloads/task6_metrics/Task6_llmmerge/Task6_InternVL3_5-8B_extracted_features_qwen_plus.csv"

CONCURRENCY = 20
TIMEOUT_SECS = 60.0
RETRIES = 3
BACKOFF_SECS = 1.5
TEMPERATURE = 0
TOP_P = 0.9
MAX_TOKENS = 8  
# =======================================================

SEMIOLOGY_FEATURES = [
    'occur_during_sleep', 'head_turning','blank_stare','close_eyes',
    'eye_blinking','face_pulling','face_twitching','tonic','clonic','arm_straightening','arm_flexion',
    'figure4','oral_automatisms','limb_automatisms','asynchronous_movement','pelvic_thrusting','full_body_shaking',
    'arms_move_simultaneously','verbal_responsiveness','ictal_vocalization'
]

def create_prompt(report_text: str, feature_name: str) -> str:
    formatted_feature = feature_name.replace('_', ' ').title()
    return f"""
You are an expert clinical neurologist. Determine if the following seizure report explicitly mentions or describes the feature: "{formatted_feature}".

[Clinical Report]
\"\"\"{(report_text or '').strip()}\"\"\"

Answer with a single word ONLY: YES or NO.
""".strip()

def parse_yes_no(content: str):
    if not content:
        return "NA"
    head = re.split(r'\s+', content.strip().upper())[0]
    if head == "YES":
        return 1
    if head == "NO":
        return 0
    return "NA"
# =================================================================

async def call_qwen_yesno(
    session: aiohttp.ClientSession,
    user_prompt: str,
) -> Tuple[bool, str]:

    url = f"{API_BASE.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    body_base = {
        "model": MODEL,
        "messages": [{"role": "user", "content": user_prompt}],
        "max_tokens": MAX_TOKENS,
        "n": 1,
    }
    bodies = [
        {**body_base, "temperature": TEMPERATURE, "top_p": TOP_P},
        {**body_base}
    ]

    for attempt in range(RETRIES):
        for variant_idx, body in enumerate(bodies):
            try:
                async with session.post(url, headers=headers, json=body, timeout=TIMEOUT_SECS) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        if "Unsupported parameter" in text and variant_idx == 0:
                            continue
                        raise RuntimeError(f"HTTP {resp.status}: {text}")

                    payload = await resp.json()
                    text = (payload.get("choices", [{}])[0]
                                   .get("message", {})
                                   .get("content", "") or "").strip()
                    text = re.sub(r'^[\"“”`]+|[\"“”`]+$', '', text).strip()
                    text = re.sub(r'\s+', ' ', text)
                    return True, text
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
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
    if not isinstance(report_text, str) or not report_text.strip():
        return row_index, feature, "NA"

    prompt = create_prompt(report_text, feature)

    async with sem:
        ok, text = await call_qwen_yesno(session, prompt)

    if not ok:
        return row_index, feature, "NA"

    return row_index, feature, parse_yes_no(text)

async def main_async():
    if not API_KEY:
        raise SystemExit("DASHSCOPE_API_KEY is empty. Please set it in environment variables.")

    try:
        df = pd.read_csv(INFERENCE_FILE)
    except FileNotFoundError:
        raise SystemExit(f"Input file not found: {INFERENCE_FILE}")

    if "file_name" not in df.columns or "report" not in df.columns:
        raise SystemExit("Input CSV must contain columns: 'file_name' and 'report'.")

    df_out = pd.DataFrame({"file_name": df["file_name"]})
    for f in SEMIOLOGY_FEATURES:
        df_out[f] = "NA"  

    tasks = []
    sem = asyncio.Semaphore(CONCURRENCY)
    async with aiohttp.ClientSession() as session:
        for i, row in df.iterrows():
            report_text = row.get("report", "")
            for feature in SEMIOLOGY_FEATURES:
                tasks.append(
                    judge_one_feature_task(sem, session, i, feature, report_text)
                )

        total = len(tasks)
        print(f"[Info] Total feature-judgement calls: {total} "
              f"({len(df)} reports × {len(SEMIOLOGY_FEATURES)} features)")

        results = []
        for chunk_start in range(0, total, CONCURRENCY * 5):
            chunk = tasks[chunk_start: chunk_start + CONCURRENCY * 5]
            results.extend(await asyncio.gather(*chunk))
            print(f"[Progress] {min(len(results), total)}/{total} done")

    for row_index, feature, val in results:
        df_out.at[row_index, feature] = val  

    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"[Done] Wrote: {OUTPUT_FILE}")
    print("\nPreview:")
    print(df_out.head())

if __name__ == "__main__":
    asyncio.run(main_async())

