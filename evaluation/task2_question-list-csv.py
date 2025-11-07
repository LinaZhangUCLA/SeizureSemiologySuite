#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, re, csv, argparse, asyncio, time, json
import pandas as pd
import backoff
from typing import List, Dict, Any, Tuple, Optional, OrderedDict
from collections import OrderedDict as OD
from openai import AsyncOpenAI
from openai import BadRequestError

# =========================
# Allowed vocabularies
# =========================
ALLOWED_PARTS = [
    "head", "eye", "mouth", "face", "arm", "leg", "full body"
]

ALLOWED_SIDES = [
    "head","mouth","full body",
    "left eye", "right eye", "both eyes",
    "left face", "right face", "whole face",
    "left arm", "right arm", "both arms",
    "left leg", "right leg", "both legs",
]


# ====== Light sanitization for moderation retry ======
_SENSITIVE_SUBS = [
    (re.compile(r'\b(sex|sexual|intercourse)\b', re.I), 'activity'),
    (re.compile(r'\b(genitals?|penis|vagina|breasts?|nipples?)\b', re.I), 'body area'),
    (re.compile(r'\b(butt(?:ock)?s?|groin|crotch)\b', re.I), 'body area'),
    (re.compile(r'\b(thrust(?:ing)?)\b', re.I), 'pushing'),
]

def sanitize_justification(text: str) -> str:
    if not text:
        return text
    s = str(text)
    for pat, rep in _SENSITIVE_SUBS:
        s = pat.sub(rep, s)
    return s

# =========================
# Question list 
# =========================
QUESTIONS: List[Dict[str, Any]] = [
    {"id":"occur_during_sleep_Q1","text":"Is the patient sleeping at the beginning of the video?","format":"yn","tag":"occur_during_sleep"},

    {"id":"head_turning_Q1","text":"Does the head turn forcefully or stiffly relative to the shoulders to one side for at least a few seconds?","format":"yn","tag":"head_turning"},
    {"id":"head_turning_Q2","text":"Does the head turn to the patient's left or to the patient's right?","format":"lr","tag":"head_turning"},

    {"id":"blank_stare_Q1","text":"Are the eyes open?","format":"yn","tag":"blank_stare"},
    {"id":"blank_stare_Q2","text":"Do the pupils of the eyes remain in a fixed position?","format":"yn","tag":"blank_stare"},

    {"id":"close_eyes_Q1","text":"Does the closure appear forceful, as if being squeezed shut?","format":"yn","tag":"close_eyes"},

    {"id":"eye_blinking_Q1","text":"Is the blinking faster than 1Hz or more?","format":"yn","tag":"eye_blinking"},
    {"id":"eye_blinking_Q2","text":"Does the blinking happen for at least a few seconds?","format":"yn","tag":"eye_blinking"},

    {"id":"face_pulling_Q1","text":"Does the patient exhibit unilateral sustained face-pulling movements?","format":"yn","tag":"face_pulling"},
    {"id":"face_pulling_Q2","text":"Does the face pull to the patient’s left or to the patient’s right?","format":"lr","tag":"face_pulling"},
    {"id":"face_pulling_Q3","text":"Does the face pulling sustain for at least a few seconds?","format":"yn","tag":"face_pulling"},

    {"id":"face_twitching_Q1","text":"Are the twitches brief and repetitive?","format":"yn","tag":"face_twitching"},
    {"id":"face_twitching_Q2","text":"Does the face twitch to the patient’s left or to the patient’s right? (L/R)","format":"lr","tag":"face_twitching"},

    {"id":"tonic_Q1","text":"Does the patient exhibit sudden muscle stiffness or rigidity?","format":"yn","tag":"tonic"},
    {"id":"tonic_Q2","text":"Which body parts are affected?","format":"choice_multi","tag":"tonic","options_parts": ALLOWED_PARTS},
    {"id":"tonic_Q3","text":"For each affected body part, specify which side of the patient is involved? Select all that apply (sides or whole)","format":"choice_multi","tag":"tonic","options_sides": ALLOWED_SIDES},

    {"id":"clonic_Q1","text":"Are the movements rhythmic, stereotyped and repetitive jerks?","format":"yn","tag":"clonic"},
    {"id":"clonic_Q2","text":"Which body parts are affected?","format":"choice_multi","tag":"clonic","options_parts": ALLOWED_PARTS},
    {"id":"clonic_Q3","text":"For each affected body part, specify which side of the patient is involved? Select all that apply (sides or whole).","format":"choice_multi","tag":"clonic","options_sides": ALLOWED_SIDES},

    {"id":"arm_flexion_Q1","text":"Are arms held in a flexed posture?","format":"yn","tag":"arm_flexion"},
    {"id":"arm_flexion_Q2","text":"How many arms are held in a flexed posture?","format":"choice","tag":"arm_flexion","options":["one","two"]},
    {"id":"arm_flexion_Q3","text":"Is this posture sustained?","format":"yn","tag":"arm_flexion"},
    {"id":"arm_flexion_Q4","text":"For each affected arm, specify which side of the patient is involved?","format":"choice","tag":"arm_flexion","options":["left arm","right arm","both arms"]},

    {"id":"arm_straightening_Q1","text":"Are arms held in an extended posture?","format":"yn","tag":"arm_straightening"},
    {"id":"arm_straightening_Q2","text":"How many arms are held in an extended posture?","format":"choice","tag":"arm_straightening","options":["one","two"]},
    {"id":"arm_straightening_Q3","text":"Is this posture sustained?","format":"yn","tag":"arm_straightening"},
    {"id":"arm_straightening_Q4","text":"For each affected arm, specify which side of the patient is involved?","format":"choice","tag":"arm_straightening","options":["left arm","right arm","both arms"]},

    {"id":"figure4_Q1","text":"Is one arm extended while the other is flexed?","format":"yn","tag":"figure4"},
    {"id":"figure4_Q2","text":"Which arm of the patient is extended and which arm is flexed?","format":"choice","tag":"figure4","options":["left arm extended right arm flexed","right arm extended left arm flexed"]},
    {"id":"figure4_Q3","text":"Is this posture sustained?","format":"yn","tag":"figure4"},

    {"id":"oral_automatisms_Q1","text":"Does the patient have repetitive, stereotyped mouth or lip movements?","format":"yn","tag":"oral_automatisms"},

    {"id":"limb_automatisms_Q1","text":"Does the patient have repetitive stereotyped movements with their hands or legs?","format":"yn","tag":"limb_automatisms"},
    {"id":"limb_automatisms_Q2","text":"For each involved limb, specify which side of the patient is involved?","format":"choice_multi","tag":"limb_automatisms","options":["left arm","right arm","both arms","left leg","right leg","both legs"]},

    {"id":"pelvic_thrusting_Q1","text":"Are there repetitive anterior posterior movements of the hips?","format":"yn","tag":"pelvic_thrusting"},

    {"id":"full_body_shaking_Q1","text":"Does the patient experience shaking of the entire body, including arms, legs, and torso?","format":"yn","tag":"full_body_shaking"},

    {"id":"asynchronous_movement_Q1","text":"Are the limbs of the patient shaking?","format":"yn","tag":"asynchronous_movement"},
    {"id":"asynchronous_movement_Q2","text":"Which limbs of the patient are shaking? (list parts)","format":"choice_multi","tag":"asynchronous_movement","options":["left arm","right arm","arms","left leg","right leg","legs","full body"]},
    {"id":"asynchronous_movement_Q3","text":"Are the limbs shaking at variable frequency, amplitude or both?","format":"choice","tag":"asynchronous_movement","options":["frequency","amplitude","both","neither"]},
    {"id":"asynchronous_movement_Q4","text":"Do the movements appear non-stereotyped?","format":"yn","tag":"asynchronous_movement"},
    {"id":"asynchronous_movement_Q5","text":"Are the limb movements asynchronous with respect to one another?","format":"yn","tag":"asynchronous_movement"},

    {"id":"arms_move_simultaneously_Q1","text":"Did the arms begin moving approximately at the same time?","format":"yn","tag":"arms_move_simultaneously"},

    {"id":"verbal_responsiveness_Q1","text":"Did anyone address the patient verbally?","format":"yn","tag":"verbal_responsiveness"},
    {"id":"verbal_responsiveness_Q2","text":"If the patient was addressed verbally, did the patient provide a coherent answer?","format":"yn","tag":"verbal_responsiveness"},

    {"id":"ictal_vocalization_Q1","text":"Is the patient making groaning, moaning, or guttural sounds?","format":"yn","tag":"ictal_vocalization"},
    {"id":"ictal_vocalization_Q2","text":"Are the noises meaningless, stereotyped, and repetitive?","format":"yn","tag":"ictal_vocalization"},
]

# =========================
# justification column mapping
# =========================
JUST_COL_BY_TAG: Dict[str,str] = {
    "occur_during_sleep": "justification_for_occur_during_sleep",
    "head_turning":"justification_for_head_turning",
    "blank_stare":"justification_for_blank_stare",
    "close_eyes":"justification_for_close_eyes",
    "eye_blinking":"justification_for_eye_blinking",
    "face_pulling":"justification_for_face_pulling",
    "face_twitching":"justification_for_face_twitching",
    "tonic":"justification_for_tonic",
    "clonic":"justification_for_clonic",
    "arm_straightening":"justification_for_arm_straightening",
    "arm_flexion":"justification_for_arm_flexion",
    "figure4":"justification_for_figure4",
    "oral_automatisms":"justification_for_oral_automatisms",
    "limb_automatisms":"justification_for_limb_automatisms",
    "asynchronous_movement":"justification_for_asynchronous_movement",
    "pelvic_thrusting":"justification_for_pelvic_thrusting",
    "full_body_shaking":"justification_for_full_body_shaking",
    "arms_move_simultaneously":"justification_for_arms_move_simultaneously",
    "verbal_responsiveness":"justification_for_verbal_responsiveness",
    "ictal_vocalization":"justification_for_ictal_vocalization",
}

# =========================
# Build bank by tag
# =========================
def build_question_bank(qs: List[Dict[str,Any]]) -> "OrderedDict[str, List[Dict[str,Any]]]":
    bank = OD()
    for q in qs:
        bank.setdefault(q["tag"], []).append(q)
    return bank

QUESTION_BANK = build_question_bank(QUESTIONS)

# =========================
# Prompt templates (per-tag single call)
# =========================
SYSTEM_PROMPT = (
    "Return ONLY the JSON object requested. No explanations. "
    "All answers must follow exactly the allowed formats and options, in lowercase. "
    "For left/right questions, if side is not determinable, answer 'neither'. "
    "For choice/choice_multi, if none of the provided options apply or cannot be determined, answer 'unknown'. "
    "Never invent information not supported by the given text. "
)

FEATURE_TMPL = """You are given a short text snippet. Based only on this text, answer all questions for the feature tag "{tag}" in ONE JSON object.

Text:
{justification}

Questions (answer each independently UNLESS explicitly constrained):
{questions_block}

Rules:
- Deterministic answers.
- Use lowercase only.
- Allowed formats:
  - yn: "yes" or "no". 
  - lr: "left" or "right" or "neither".
  - choice: choose exactly one item from the provided options. If none apply or cannot be determined, answer "unknown".
  - choice_multi: choose a COMMA-SEPARATED subset, and each item of the subset must be chosen from the provided options. If none apply or cannot be determined, answer "unknown".

Output JSON schema (print ONLY this JSON, no backticks, no extra text):
{{
{schema_body}
}}
"""

def build_feature_prompt(tag: str, questions: List[Dict[str,Any]], justification: str) -> str:
    qlines = []
    schema_lines = []
    for q in questions:
        qid = q["id"]; fmt = q["format"]; text = q["text"]
        if fmt == "yn":
            qlines.append(f'- {qid} (yn): {text}')
            schema_lines.append(f'  "{qid}": "yes|no"')
        elif fmt == "lr":
            qlines.append(f'- {qid} (lr): {text}')
            schema_lines.append(f'  "{qid}": "left|right|neither"')
        elif fmt == "choice":
            opts = [o.lower() for o in (q.get("options") or q.get("options_parts") or q.get("options_sides") or [])]
            qlines.append(f'- {qid} (choice): {text}  Options: {", ".join(opts)}')
            schema_lines.append(f'  "{qid}": ONE OF [{", ".join(opts + ["unknown"])}]')

        elif fmt == "choice_multi":
            opts = [o.lower() for o in (q.get("options") or q.get("options_parts") or q.get("options_sides") or ALLOWED_PARTS)]
            qlines.append(f'- {qid} (choice_multi): {text}  Options: {", ".join(opts)}  (comma-separated; or "unknown")')
            schema_lines.append(f'  "{qid}": COMMA-LIST of subset of [{", ".join(opts)}] OR "unknown"')        
        else:
            qlines.append(f'- {qid} (text): {text}')
            schema_lines.append(f'  "{qid}": "unknown"')
    questions_block = "\n".join(qlines)
    schema_body = ",\n".join(schema_lines)
    return FEATURE_TMPL.format(
        tag=tag,
        justification=justification,
        questions_block=questions_block,
        schema_body=schema_body
    )

# =========================
# Normalizers
# =========================
_LABEL_RE = re.compile(r"(?i)\b(yn|lr|choice|choice_multi|pair|list|map|text)\s*[:=]\s*")

def _clean_raw(x: str) -> str:
    s = (x or "").strip()
    s = _LABEL_RE.sub("", s)
    s = s.replace("|", ":").replace("，", ",").replace("；", ";")
    s = s.replace("\r", "\n").strip()
    return s

def norm_text(x: str) -> str:
    s = _clean_raw(x).lower()
    s = re.sub(r"[\s\n\t]+", " ", s).strip()
    return s if s else "unknown"
def norm_yn(x: str) -> str:
    s = norm_text(x)
    if s in {"yes","no"}: 
        return s
    return "no"
def norm_lr(x: str) -> str:
    s = norm_text(x)
    if s in {"left","right"}: 
        return s
    if s in {"l","r"}: 
        return "left" if s=="l" else "right"
    return "neither"
def norm_choice(x: str, options: List[str]) -> str:
    s = norm_text(x)
    opts = set(o.lower() for o in options + ["unknown"])
    return s if s in opts else "unknown"
def norm_choice_multi(x: str, allowed_items: List[str]) -> str:
    s = _clean_raw(x).lower().strip()
    if not s or s == "unknown":
        return "unknown"

    tokens = [i.strip() for i in re.split(r"[,\n;]+", s) if i.strip()]
    allow = [a.lower() for a in allowed_items]
    allow_set = set(allow)

    kept = [it for it in tokens if it in allow_set]

    if not kept:
        return "unknown"

    order = {a: idx for idx, a in enumerate(allow)}
    kept = sorted(list(dict.fromkeys(kept)), key=lambda z: order.get(z, 10**9))
    return ", ".join(kept)

def default_for(fmt: str) -> str:
    return "unknown"

# =========================
# Single-call per tag
# =========================
@backoff.on_exception(backoff.expo, Exception, max_time=120)
async def ask_feature_once(client, model: str, tag: str, justification: str, qs: List[Dict[str,Any]], dry: bool, file_name: str = "") -> Dict[str,str]:
    if dry:
        return {q["id"]: "unknown" for q in qs}

    # using the justification for prompt
    prompt = build_feature_prompt(tag, qs, justification)

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":SYSTEM_PROMPT},
                      {"role":"user","content":prompt}],
            temperature=0,
        )
        txt = (resp.choices[0].message.content or "").strip()

    except BadRequestError as e:
        if "data_inspection_failed" in str(e):
            safe_just = sanitize_justification(justification)
            safe_prompt = build_feature_prompt(tag, qs, safe_just)
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role":"system","content":SYSTEM_PROMPT},
                              {"role":"user","content":safe_prompt}],
                    temperature=0,
                )
                txt = (resp.choices[0].message.content or "").strip()
            except Exception:
                return {q["id"]: "unknown" for q in qs}
        else:
            return {q["id"]: "unknown" for q in qs}

    try:
        data = json.loads(txt)
    except Exception:
        txt2 = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt.strip(), flags=re.I|re.M)
        data = json.loads(txt2)

    out: Dict[str,str] = {}
    for q in qs:
        qid, fmt = q["id"], q["format"]
        raw = str(data.get(qid, "")).strip()
        if fmt == "yn":
            out[qid] = norm_yn(raw)
        elif fmt == "lr":
            out[qid] = norm_lr(raw)
        elif fmt == "choice":
            out[qid] = norm_choice(raw, [o.lower() for o in (q.get("options") or q.get("options_parts") or q.get("options_sides") or [])])
        elif fmt == "choice_multi":
            out[qid] = norm_choice_multi(raw, [o.lower() for o in (q.get("options") or q.get("options_parts") or q.get("options_sides") or ALLOWED_PARTS)])
        else:
            out[qid] = norm_text(raw)

    return out


def _get_q1(qs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for q in qs:
        if q.get("format") == "yn" and q.get("id", "").endswith("_Q1"):
            return q
    return None

@backoff.on_exception(backoff.expo, Exception, max_time=120)
async def ask_feature_sequential(client, model: str, tag: str, justification: str,
                                 qs: List[Dict[str,Any]], dry: bool,
                                 file_name: str = "") -> Dict[str, str]:
    out: Dict[str, str] = {}

    q1 = _get_q1(qs)
    if q1 is None:
        return await ask_feature_once(client, model, tag, justification, qs, dry, file_name=file_name)

    if (justification or "").strip() == "":
        out[q1["id"]] = "no"
        for q in qs:
            if q["id"] != q1["id"]:
                out[q["id"]] = "none"
        return out

    ans_q1 = await ask_feature_once(client, model, tag, justification, [q1], dry, file_name=file_name)
    q1_ans = ans_q1.get(q1["id"], "no")  
    out.update(ans_q1)

    if q1_ans == "no":
        for q in qs:
            if q["id"] != q1["id"]:
                out[q["id"]] = "none"
        return out

    rest_qs = [q for q in qs if q["id"] != q1["id"]]
    if rest_qs:
        ans_rest = await ask_feature_once(client, model, tag, justification, rest_qs, dry, file_name=file_name)
        out.update(ans_rest)
    return out


# =========================
# Row processor (loop tags)
# =========================
async def process_row(row_idx: int, row: pd.Series, client, model: str, dry: bool) -> Dict[str,str]:
    out: Dict[str,str] = {}
    for tag, qs in QUESTION_BANK.items():
        just_col = JUST_COL_BY_TAG.get(tag, None)
        justification = ""
        if just_col and just_col in row.index:
            justification = str(row.get(just_col) or "").strip()

        if not justification:
            continue

        ans_map = await ask_feature_sequential(client, model, tag, justification, qs, dry, file_name=row.iloc[0])
        out.update(ans_map)

    return out

#async def process_row(row_idx: int, row: pd.Series, client, model: str, dry: bool) -> Dict[str,str]:
#    out: Dict[str,str] = {}
#    for tag, qs in QUESTION_BANK.items():
#        just_col = JUST_COL_BY_TAG.get(tag, None)
#        justification = str(row.get(just_col, "") or "") if just_col and just_col in row.index else ""
#        ans_map = await ask_feature_sequential(client, model, tag, justification, qs, dry, file_name=row.iloc[0])
#
#        out.update(ans_map)
#
#    # ensure all columns exist
#    for q in QUESTIONS:
#        if q["id"] not in out:
#            out[q["id"]] = "unknown"
#    return out

# =========================
# CSV helpers (resume & write)
# =========================
def load_done(output_csv: str) -> Tuple[set, Optional[pd.DataFrame]]:
    if not os.path.exists(output_csv):
        return set(), None
    try:
        done_df = pd.read_csv(output_csv, dtype=str)
        if done_df.empty:
            return set(), None
        first_col = done_df.columns[0]
        done = set(done_df[first_col].astype(str).tolist())
        return done, done_df
    except Exception:
        return set(), None

#def ensure_output_columns(df: pd.DataFrame, first_col: str) -> pd.DataFrame:
#    cols = [first_col] + [q["id"] for q in QUESTIONS]
#    for c in cols:
#        if c not in df.columns:
#            df[c] = "unknown"
#   return df[cols]

def concat_and_save(done_df: Optional[pd.DataFrame],
                    new_rows: List[Dict[str,str]],
                    first_col: str,
                    out_csv: str,
                    order_list: List[str]):

    new_df = pd.DataFrame(new_rows)

    if new_df.empty:
        new_df = pd.DataFrame(columns=[first_col])

    def ordered_cols(df: pd.DataFrame) -> List[str]:
        qids_in_order = [q["id"] for q in QUESTIONS]
        present_qids = [qid for qid in qids_in_order if qid in df.columns]
        others = [c for c in df.columns if c not in ([first_col] + present_qids)]
        return [first_col] + present_qids + others

    if done_df is None:
        cols = ordered_cols(new_df)
        final_df = new_df.reindex(columns=cols)
    else:
        tmp_df = pd.concat([done_df, new_df], axis=0, ignore_index=True, sort=False)
        cols = ordered_cols(tmp_df)
        final_df = tmp_df.reindex(columns=cols)
        final_df = final_df.drop_duplicates(subset=[first_col], keep="first")

    present = set(final_df[first_col].astype(str).tolist())
    ordered_keys = [k for k in order_list if k in present]
    final_df = final_df.set_index(first_col).loc[ordered_keys].reset_index()

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    tmp = out_csv + ".tmp"
    final_df.to_csv(tmp, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(tmp, out_csv)

# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Per-tag single-call extraction to wide CSV (Q2->Q3 constraint; controlled options)."
    )
    ap.add_argument("--input-csv",  required=True, help="Input CSV (has justification_for_* columns).")
    ap.add_argument("--output-csv", required=True, help="Output CSV path.")
    ap.add_argument("--api-key",    default=os.environ.get("DASHSCOPE_API_KEY"), help="DashScope key; or set env DASHSCOPE_API_KEY.")
    ap.add_argument("--model",      default="qwen-plus")
    ap.add_argument("--base-url",   default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    ap.add_argument("--concurrency",type=int, default=10, help="Rows (videos) processed concurrently.")
    ap.add_argument("--limit",      type=int, default=0, help="If >0, only run first N rows (after de-dup).")
    ap.add_argument("--dry-run",    action="store_true", help="No API calls; fill defaults.")
    return ap.parse_args()

# =========================
# Main
# =========================
async def main():
    args = parse_args()
    if not os.path.isfile(args.input_csv):
        raise FileNotFoundError(args.input_csv)
    if (not args.dry_run) and (not args.api_key):
        raise RuntimeError("Missing API key. Pass --api-key or set env DASHSCOPE_API_KEY.")

    df = pd.read_csv(args.input_csv, dtype=str)
    first_col = df.columns[0]
    df[first_col] = df[first_col].astype(str)
    df = df.drop_duplicates(subset=[first_col], keep="first").reset_index(drop=True)
    if args.limit and args.limit > 0:
        df = df.iloc[:args.limit, :]
    
    order_list = df[first_col].tolist()

    done_set, done_df = load_done(args.output_csv)
    todo = df[~df[first_col].isin(done_set)].reset_index(drop=True)

    client = None if args.dry_run else AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)

    sem = asyncio.Semaphore(args.concurrency)
    new_rows: List[Dict[str,str]] = []
    total = len(todo)

    async def worker(i: int, row: pd.Series):
        async with sem:
            fn = row[first_col]
            t0 = time.time()
            ans_map = await process_row(i, row, client, args.model, args.dry_run)
            rec = {first_col: fn}; rec.update(ans_map)
            dt = time.time() - t0
            print(f"[{i+1}/{total}] {fn} done in {dt:.2f}s")
            return rec

    tasks = [asyncio.create_task(worker(i, row)) for i, row in todo.iterrows()]
    batch = []
    for k, task in enumerate(asyncio.as_completed(tasks), 1):
        rec = await task
        batch.append(rec)
        if (k % 20 == 0) or (k == total):
            concat_and_save(done_df, batch, first_col, args.output_csv, order_list)
            try:
                done_df = pd.read_csv(args.output_csv, dtype=str)
            except Exception:
                done_df = None
            batch = []

    if total == 0 and done_df is not None:
        present = set(done_df[first_col].astype(str).tolist())
        ordered_keys = [k for k in order_list if k in present]

        qids_in_order = [q["id"] for q in QUESTIONS]
        present_qids = [qid for qid in qids_in_order if qid in done_df.columns]
        others = [c for c in done_df.columns if c not in ([first_col] + present_qids)]
        cols = [first_col] + present_qids + others

        done_df = done_df.reindex(columns=cols)
        done_df = done_df.set_index(first_col).loc[ordered_keys].reset_index()
        done_df.to_csv(args.output_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Saved -> {args.output_csv}")

if __name__ == "__main__":
    asyncio.run(main())
