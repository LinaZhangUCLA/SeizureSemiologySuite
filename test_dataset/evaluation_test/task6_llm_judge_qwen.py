import os
import re
import json
import time
import asyncio
import aiohttp
import pandas as pd
from typing import Optional, Tuple, List
from tqdm import tqdm

# ===================== Qwen API settings =====================
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY  = os.environ.get("DASHSCOPE_API_KEY", "").strip()
MODEL    = "qwen-plus"

# ===================== File paths =====================
# Ground-truth (human) reports CSV — must contain: file_name, report
REF_CSV = "result/ground_truth/task6_report_annotation.csv"

# Inference (AI) reports CSV — must contain: video_name, report
CSV_IN  = "result/vlm_inference/Qwen2.5-VL-7B-Instruct/Task6_Qwen2.5-VL-7B-Instruct_all_merged_llmmerge.csv"

# Output CSV
CSV_OUT = "metrics/Task6_llmmerge_new/Task6_Qwen2.5-VL-7B-Instruct_similarity_scores_async.csv"


# ===================== Async settings =====================
CONCURRENCY  = 16
TIMEOUT_SECS = 60.0
RETRIES      = 3
BACKOFF_SECS = 1.5
TEMPERATURE = 0
TOP_P       = 0.9

# ===================== Helpers =====================
def extract_base_filename(video_name: str) -> str:
    """
    Extract base filename from video_name by removing _segment_X suffix
    e.g., 'A0002@5-13-2021@UA6693LK@sz_v1_1_segment_0.mp4' -> 'A0002@5-13-2021@UA6693LK@sz_v1_1.mp4'
    """
    if not isinstance(video_name, str):
        return str(video_name)
    # Remove _segment_X pattern
    return re.sub(r'_segment_\d+', '', video_name)

# ===================== Prompt =====================
def create_prompt(reference_report: str, inference_report: str) -> str:
    """Ask for a single float similarity score in [0.0, 1.0]."""
    return f"""
You are an expert clinical neurologist evaluating an AI-generated report of a patient's seizure episode. Your task is to assess the semantic and clinical similarity of the AI's report compared to a ground truth report written by a human expert.

Please provide a single floating-point score from 0.0 to 1.0 based on the following criteria:
- 1.0 (Excellent): The AI-generated report accurately captures all critical semiological features mentioned in the ground truth. It is factually consistent and contains no hallucinations.
- 0.8 (Good): The AI-generated report captures the majority of the key semiological features but may miss minor details or use less precise clinical terminology.
- 0.6 (Moderate): The AI-generated report identifies some correct semiological features but misses significant ones or contains minor factual inaccuracies.
- 0.4 (Poor): The AI-generated report only describes general, non-specific events and fails to identify most of the crucial clinical signs mentioned in the ground truth.
- 0.2 (Failure): The AI-generated report is completely irrelevant, factually incorrect, or hallucinates information not supported by the ground truth.

Evaluate the following two reports:

[Ground Truth Report]
\"\"\"{(reference_report or '').strip()}\"\"\"\n
[AI-Generated Report]
\"\"\"{(inference_report or '').strip()}\"\"\"\n
Return ONLY the numerical score (a single float in [0.0, 1.0]) and nothing else.
""".strip()

# ===================== Score parsing =====================
_float_regex = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def parse_score(text: str) -> Optional[float]:
    """Parse first float-like token and clamp to [0,1]."""
    if not text:
        return None
    m = _float_regex.search(text.strip())
    if not m:
        return None
    try:
        val = float(m.group(0))
        if val < 0.0: val = 0.0
        if val > 1.0: val = 1.0
        return val
    except Exception:
        return None

# ===================== Qwen caller (async) =====================
async def call_qwen_score(
    session: aiohttp.ClientSession,
    prompt: str,
    retries: int = RETRIES,
    timeout: float = TIMEOUT_SECS,
) -> Optional[float]:
    """
    Call DashScope-compatible /chat/completions and return float score (or None).
    Tries with (temperature/top_p) first, then falls back to a body without those params
    if the API reports they are unsupported.
    """
    if not API_KEY or API_KEY == "YOUR_API_KEY":
        return None

    url = f"{API_BASE.rstrip('/')}/chat/completions"
    base_body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8,  # numeric only
    }
    # First try with TEMPERATURE/TOP_P, then a clean fallback if needed
    bodies: List[dict] = [
        {**base_body, "temperature": TEMPERATURE, "top_p": TOP_P},
        {**base_body},
    ]

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    for attempt in range(retries):
        for variant_idx, body in enumerate(bodies):
            try:
                async with session.post(url, headers=headers, json=body, timeout=timeout) as resp:
                    txt = await resp.text()
                    if resp.status != 200:
                        if "Unsupported parameter" in txt and variant_idx == 0:
                            # sampling params not supported -> try fallback body immediately
                            continue
                        if resp.status == 429 and attempt < retries - 1:
                            await asyncio.sleep(BACKOFF_SECS * (attempt + 1))
                            continue
                        raise RuntimeError(f"HTTP {resp.status}: {txt}")

                    data = json.loads(txt)
                    content = (
                        data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "") or ""
                    ).strip()
                    content = re.sub(r'^[\"""`]+|[\"""`]+$', '', content).strip()
                    content = re.sub(r"\s+", " ", content)
                    score = parse_score(content)
                    if score is not None:
                        return score
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError):
                pass

        if attempt < retries - 1:
            await asyncio.sleep(BACKOFF_SECS * (attempt + 1))

    return None

# ===================== Per-row task =====================
async def score_one_row(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    idx: int,
    reference_report: str,
    inference_report: str,
) -> Tuple[int, Optional[float]]:
    """Score a single pair (reference, inference)."""
    if not isinstance(reference_report, str) or not reference_report.strip():
        return idx, None
    if not isinstance(inference_report, str) or not inference_report.strip():
        return idx, None
    prompt = create_prompt(reference_report, inference_report)
    async with sem:
        score = await call_qwen_score(session, prompt)
    return idx, score

# ===================== Orchestrator =====================
async def main_async():
    if not API_KEY or API_KEY == "YOUR_API_KEY":
        print("Missing DashScope API key. Set API_KEY (or env DASHSCOPE_API_KEY).")
        raise SystemExit(1)

    # 1) Load CSVs
    try:
        df_ref = pd.read_csv(REF_CSV)
        df_inf = pd.read_csv(CSV_IN)
    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
        raise SystemExit(1)

    # 2) Check required columns
    if "file_name" not in df_ref.columns:
        print(f"Reference CSV must contain 'file_name' column. Found: {df_ref.columns.tolist()}")
        raise SystemExit(1)
    
    if "video_name" not in df_inf.columns:
        print(f"Inference CSV must contain 'video_name' column. Found: {df_inf.columns.tolist()}")
        raise SystemExit(1)
    
    if "report" not in df_ref.columns or "report" not in df_inf.columns:
        print("Both CSVs must contain a 'report' column.")
        raise SystemExit(1)

    # 3) Extract base filenames from video_name (remove _segment_X suffix)
    df_inf['base_file_name'] = df_inf['video_name'].apply(extract_base_filename)
    
    print(f"Reference CSV: {len(df_ref)} rows with 'file_name'")
    print(f"Inference CSV: {len(df_inf)} rows with 'video_name'")
    print(f"Example mapping: {df_inf['video_name'].iloc[0]} -> {df_inf['base_file_name'].iloc[0]}")

    # 4) Rename and merge
    df_ref = df_ref.rename(columns={"report": "reference_report", "file_name": "base_file_name"})
    df_inf = df_inf.rename(columns={"report": "inference_report"})

    # Merge on base_file_name
    df = pd.merge(df_ref[['base_file_name', 'reference_report']], 
                  df_inf[['base_file_name', 'video_name', 'inference_report']], 
                  on="base_file_name", 
                  how="inner")
    
    n = len(df)
    if n == 0:
        print("No matching records found after merge. Check if file names match correctly.")
        raise SystemExit(1)
    
    print(f"Starting similarity score evaluation for {n} records...")

    # 5) Build tasks
    sem = asyncio.Semaphore(CONCURRENCY)
    scores: List[Optional[float]] = [None] * n

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=TIMEOUT_SECS, sock_read=TIMEOUT_SECS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            score_one_row(
                sem, session, i,
                df.iloc[i]["reference_report"],
                df.iloc[i]["inference_report"],
            )
            for i in range(n)
        ]
        for fut in tqdm(asyncio.as_completed(tasks), total=n):
            idx, score = await fut
            scores[idx] = score

    # 6) Append final average row (model-level score)
    df["similarity_score"] = scores
    avg_score = pd.to_numeric(df["similarity_score"], errors="coerce").mean()  # skips NaN
    
    summary_row = {
        "base_file_name": "MODEL_AVG",
        "video_name": "",
        "reference_report": "",
        "inference_report": "",
        "similarity_score": avg_score,
    }
    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

    # 7) Save
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    df.to_csv(CSV_OUT, index=False)
    print(f"\nSimilarity evaluation complete. Results saved to {CSV_OUT}")
    print(f"Model average score: {avg_score:.4f}" if pd.notna(avg_score) else "Model average score: NA")
    print("\nPreview (last 5 rows):")
    print(df[["base_file_name", "video_name", "similarity_score"]].tail(5))

if __name__ == "__main__":
    asyncio.run(main_async())


#previous
# import os
# import re
# import json
# import time
# import asyncio
# import aiohttp
# import pandas as pd
# from typing import Optional, Tuple, List
# from tqdm import tqdm

# # ===================== Qwen API settings =====================
# API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# API_KEY  = "API key"  # <-- Put your DashScope API key here
# MODEL    = "qwen-plus"

# # ===================== File paths =====================
# # Ground-truth (human) reports CSV — must contain: file_name, report
# REF_CSV = "/Users/pxy_amber/Downloads/task6_metrics/task6_annotation_new.csv"

# # Inference (AI) reports CSV — must contain: file_name, report
# CSV_IN  = "/Users/pxy_amber/Downloads/task6_metrics/Task6_llmmerge/Task6_InternVL3_5-8B_all_merged_llmmerge.csv"

# # Output CSV
# CSV_OUT = "/Users/pxy_amber/Downloads/task6_metrics/Task6_llmmerge/Task6_InternVL3_5-8B_similarity_scores_async.csv"


# # ===================== Async settings =====================
# CONCURRENCY  = 16
# TIMEOUT_SECS = 60.0
# RETRIES      = 3
# BACKOFF_SECS = 1.5
# TEMPERATURE = 0
# TOP_P       = 0.9

# # ===================== Prompt =====================
# def create_prompt(reference_report: str, inference_report: str) -> str:
#     """Ask for a single float similarity score in [0.0, 1.0]."""
#     return f"""
# You are an expert clinical neurologist evaluating an AI-generated report of a patient's seizure episode. Your task is to assess the semantic and clinical similarity of the AI's report compared to a ground truth report written by a human expert.

# Please provide a single floating-point score from 0.0 to 1.0 based on the following criteria:
# - 1.0 (Excellent): The AI-generated report accurately captures all critical semiological features mentioned in the ground truth. It is factually consistent and contains no hallucinations.
# - 0.8 (Good): The AI-generated report captures the majority of the key semiological features but may miss minor details or use less precise clinical terminology.
# - 0.6 (Moderate): The AI-generated report identifies some correct semiological features but misses significant ones or contains minor factual inaccuracies.
# - 0.4 (Poor): The AI-generated report only describes general, non-specific events and fails to identify most of the crucial clinical signs mentioned in the ground truth.
# - 0.2 (Failure): The AI-generated report is completely irrelevant, factually incorrect, or hallucinates information not supported by the ground truth.

# Evaluate the following two reports:

# [Ground Truth Report]
# \"\"\"{(reference_report or '').strip()}\"\"\"\n
# [AI-Generated Report]
# \"\"\"{(inference_report or '').strip()}\"\"\"\n
# Return ONLY the numerical score (a single float in [0.0, 1.0]) and nothing else.
# """.strip()

# # ===================== Helpers =====================
# _float_regex = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# def parse_score(text: str) -> Optional[float]:
#     """Parse first float-like token and clamp to [0,1]."""
#     if not text:
#         return None
#     m = _float_regex.search(text.strip())
#     if not m:
#         return None
#     try:
#         val = float(m.group(0))
#         if val < 0.0: val = 0.0
#         if val > 1.0: val = 1.0
#         return val
#     except Exception:
#         return None

# # ===================== Qwen caller (async) =====================
# async def call_qwen_score(
#     session: aiohttp.ClientSession,
#     prompt: str,
#     retries: int = RETRIES,
#     timeout: float = TIMEOUT_SECS,
# ) -> Optional[float]:
#     """
#     Call DashScope-compatible /chat/completions and return float score (or None).
#     Tries with (temperature/top_p) first, then falls back to a body without those params
#     if the API reports they are unsupported.
#     """
#     if not API_KEY or API_KEY == "YOUR_API_KEY":
#         return None

#     url = f"{API_BASE.rstrip('/')}/chat/completions"
#     base_body = {
#         "model": MODEL,
#         "messages": [{"role": "user", "content": prompt}],
#         "max_tokens": 8,  # numeric only
#     }
#     # First try with TEMPERATURE/TOP_P, then a clean fallback if needed
#     bodies: List[dict] = [
#         {**base_body, "temperature": TEMPERATURE, "top_p": TOP_P},
#         {**base_body},
#     ]

#     headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

#     for attempt in range(retries):
#         for variant_idx, body in enumerate(bodies):
#             try:
#                 async with session.post(url, headers=headers, json=body, timeout=timeout) as resp:
#                     txt = await resp.text()
#                     if resp.status != 200:
#                         if "Unsupported parameter" in txt and variant_idx == 0:
#                             # sampling params not supported -> try fallback body immediately
#                             continue
#                         if resp.status == 429 and attempt < retries - 1:
#                             await asyncio.sleep(BACKOFF_SECS * (attempt + 1))
#                             continue
#                         raise RuntimeError(f"HTTP {resp.status}: {txt}")

#                     data = json.loads(txt)
#                     content = (
#                         data.get("choices", [{}])[0]
#                             .get("message", {})
#                             .get("content", "") or ""
#                     ).strip()
#                     content = re.sub(r'^[\"“”`]+|[\"“”`]+$', '', content).strip()
#                     content = re.sub(r"\s+", " ", content)
#                     score = parse_score(content)
#                     if score is not None:
#                         return score
#             except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError):
#                 pass

#         if attempt < retries - 1:
#             await asyncio.sleep(BACKOFF_SECS * (attempt + 1))

#     return None

# # ===================== Per-row task =====================
# async def score_one_row(
#     sem: asyncio.Semaphore,
#     session: aiohttp.ClientSession,
#     idx: int,
#     reference_report: str,
#     inference_report: str,
# ) -> Tuple[int, Optional[float]]:
#     """Score a single pair (reference, inference)."""
#     if not isinstance(reference_report, str) or not reference_report.strip():
#         return idx, None
#     if not isinstance(inference_report, str) or not inference_report.strip():
#         return idx, None
#     prompt = create_prompt(reference_report, inference_report)
#     async with sem:
#         score = await call_qwen_score(session, prompt)
#     return idx, score

# # ===================== Orchestrator =====================
# async def main_async():
#     if not API_KEY or API_KEY == "YOUR_API_KEY":
#         print("Missing DashScope API key. Set API_KEY (or env DASHSCOPE_API_KEY).")
#         raise SystemExit(1)

#     # 1) Load & merge
#     try:
#         df_ref = pd.read_csv(REF_CSV)
#         df_inf = pd.read_csv(CSV_IN)
#     except FileNotFoundError as e:
#         print(f"File not found: {e.filename}")
#         raise SystemExit(1)

#     df_ref = df_ref.rename(columns={"report": "reference_report"})
#     df_inf = df_inf.rename(columns={"report": "inference_report"})

#     if "file_name" not in df_ref.columns or "file_name" not in df_inf.columns:
#         print("Both CSVs must contain a 'file_name' column.")
#         raise SystemExit(1)

#     df = pd.merge(df_ref, df_inf, on="file_name", how="inner")
#     n = len(df)
#     print(f"Starting similarity score evaluation for {n} records...")

#     # 2) Build tasks
#     sem = asyncio.Semaphore(CONCURRENCY)
#     scores: List[Optional[float]] = [None] * n

#     timeout = aiohttp.ClientTimeout(total=None, sock_connect=TIMEOUT_SECS, sock_read=TIMEOUT_SECS)
#     async with aiohttp.ClientSession(timeout=timeout) as session:
#         tasks = [
#             score_one_row(
#                 sem, session, i,
#                 df.iloc[i]["reference_report"],
#                 df.iloc[i]["inference_report"],
#             )
#             for i in range(n)
#         ]
#         for fut in tqdm(asyncio.as_completed(tasks), total=n):
#             idx, score = await fut
#             scores[idx] = score

#     # 3) Append final average row (model-level score)
#     df["similarity_score"] = scores
#     avg_score = pd.to_numeric(df["similarity_score"], errors="coerce").mean()  # skips NaN
#     summary_row = {
#         "file_name": "MODEL_AVG",
#         "reference_report": "",
#         "inference_report": "",
#         "similarity_score": avg_score,
#     }
#     df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

#     # 4) Save
#     df.to_csv(CSV_OUT, index=False)
#     print(f"\nSimilarity evaluation complete. Results saved to {CSV_OUT}")
#     print(f"Model average score: {avg_score:.4f}" if pd.notna(avg_score) else "Model average score: NA")
#     print("\nPreview (last 5 rows):")
#     print(df[["file_name", "similarity_score"]].tail(5))

# if __name__ == "__main__":
#     asyncio.run(main_async())
