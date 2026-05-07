#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import asyncio
import aiohttp
from collections import defaultdict
from typing import List, Dict, Tuple

import pandas as pd

API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "sk-853145a6309e48ad83e2d0cd01a155a1"
MODEL = "qwen-plus-latest"


BASE_DIR = "/home/lina/ssb/SeizureSemiologyBench/result/vlm_inference_test/"


TEMPERATURE = 0
TOP_P = 0.9
MAX_TOKENS = 1024
TIMEOUT_SECS = 60.0
CONCURRENCY = 10
RETRIES = 3
BACKOFF_SECS = 1.5

PROMPT_TEMPLATE = (
    """ The following is a list of report segments from a single observation. 
    The segments are formatted as a bulleted list below: 

    {segments_list} 

    Merge them into a single consolidated report that is de-duplicated but information-preserving. 
    Remove any duplicate or redundant sentences and merge statements with the same meaning. 
    DO NOT include any introductory phrases or sentences like \"Here is a consolidated report:\". 
    Start your response directly with the merged content. """
)


def parse_filename(file_name: str) -> Tuple[str, int]:
    file_name = file_name.strip()
    m = re.match(r"^(.*)_segment_(\d+)(\.mp4)$", file_name)
    if m:
        base = m.group(1)
        seg = int(m.group(2))
        ext = m.group(3)
        original = f"{base}{ext}"
        return original, seg
    else:
        return file_name, 0


def build_prompt(segments: List[str]) -> str:
    segments_list_str = "\n- " + "\n- ".join(segments)
    return PROMPT_TEMPLATE.format(segments_list=segments_list_str)


async def call_qwen(session: aiohttp.ClientSession, user_prompt: str) -> str:
    url = f"{API_BASE.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
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
            async with session.post(
                url, headers=headers, json=body, timeout=TIMEOUT_SECS
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {text}")
                payload = await resp.json()
                text = payload["choices"][0]["message"]["content"]
                text = (text or "").strip()
                text = re.sub(r'^[`"“”]+|[`"“”]+$', "", text).strip()
                text = re.sub(r"\s+", " ", text)
                return text
        except Exception as e:
            if attempt < RETRIES - 1:
                print(f"[WARN] call failed ({attempt+1}/{RETRIES}): {e}")
                await asyncio.sleep(BACKOFF_SECS)
            else:
                print(f"[ERROR] call failed: {e}")
                return ""


async def process_group(
    idx: int,
    total: int,
    original_name: str,
    rows: List[Tuple[int, str]],
    session: aiohttp.ClientSession,
):
    rows_sorted = sorted(rows, key=lambda x: x[0])
    #print(rows_sorted)
    reports = [r[1].strip() for r in rows_sorted if r[1] and str(r[1]).strip()]

    if len(reports) <= 1:
        merged = reports[0] if reports else ""
        print(f"[{idx+1}/{total}] {original_name}: single segment, skipped merge.")
        return {"original_file_name": original_name, "report": merged}

    prompt = build_prompt(reports)
    #print(prompt)
    merged = await call_qwen(session, prompt)
    #print(merged)
    if not merged:
        merged = " || ".join(reports)
        print(f"[{idx+1}/{total}] {original_name}: llm failed, used fallback.")
    else:
        print(f"[{idx+1}/{total}] {original_name}: merged {len(reports)} segments.")
    return {"original_file_name": original_name, "report": merged}


async def main_async(INPUT_CSV,OUTPUT_CSV):
    df = pd.read_csv(INPUT_CSV)
    if "file_name" not in df.columns or "report" not in df.columns:
        raise SystemExit("CSV must contain 'file_name' and 'report' columns")

    groups: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

    for _, row in df.iterrows():
        file_name = str(row["file_name"]).strip()
        report = "" if pd.isna(row["report"]) else str(row["report"])
        original, seg_num = parse_filename(file_name)
        #print(original,seg_num)
        groups[original].append((seg_num, report))

    total_groups = len(groups)
    #print(f"[INFO] total groups: {total_groups}")
    #print(groups)
    sem = asyncio.Semaphore(CONCURRENCY)

    async def limited(i, name, rows, session):
        async with sem:
            return await process_group(i, total_groups, name, rows, session)

    async with aiohttp.ClientSession() as session:
        tasks = [
            limited(i, original, rows, session)
            for i, (original, rows) in enumerate(groups.items())
        ]
        results = await asyncio.gather(*tasks)

    out_df = pd.DataFrame(results, columns=["original_file_name", "report"])
    out_df = out_df.rename(columns={"original_file_name": "file_name"})
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"[DONE] wrote: {OUTPUT_CSV}")


if __name__ == "__main__":
    if API_KEY == "YOUR_DASHSCOPE_API_KEY":
        raise SystemExit("Please set API_KEY first.")

    MODELS = [
        #"InternVL3_5-8B",
        # "InternVL3_5-38B",
        # # "Qwen2.5-VL-7B-Instruct",
        # # "Qwen2.5-VL-32B-Instruct",
        # # "Qwen2.5-VL-72B-Instruct",
        # "Lingshu-32B",
        # # "Qwen2.5-Omni-7B",
        # # 'Qwen3-VL-8B-Instruct',
        # # 'Qwen3-VL-32B-Instruct',
        # "Qwen3-Omni-30B-A3B-Instruct",

        'seizure_omni_sft' ,
        'seizure_omni_grpo'   
    ]    
    for model in MODELS:
        INPUT_CSV = f"{BASE_DIR}/{model}/Task6_{model}_all.csv"
        OUTPUT_CSV = f"{BASE_DIR}/{model}/Task6_{model}_all_merged.csv"    
        asyncio.run(main_async(INPUT_CSV,OUTPUT_CSV))