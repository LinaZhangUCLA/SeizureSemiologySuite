#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, re, asyncio
from typing import List, Dict, Any, Optional
from collections import OrderedDict as OD
from decimal import Decimal, ROUND_HALF_UP
from openai import AsyncOpenAI

# -----------------------------
# -----------------------------
ALLOWED_PARTS = ["head", "eye", "mouth", "face", "arm", "leg", "full body"]
ALLOWED_SIDES = [
    "head","mouth","full body",
    "left eye", "right eye", "both eyes",
    "left face", "right face", "whole face",
    "left arm", "right arm", "both arms",
    "left leg", "right leg", "both legs",
]

FEATURE_QUESTIONS: Dict[str, List[Dict[str, Any]]] = OD([
    ("occur_during_sleep", [
        {"id":"occur_during_sleep_Q1","text":"Is the patient sleeping at the beginning of the video?","format":"yn"},
    ]),

    ("head_turning", [
        {"id":"head_turning_Q1","text":"Does the head turn forcefully or stiffly relative to the shoulders to one side for at least a few seconds?","format":"yn"},
        {"id":"head_turning_Q2","text":"Does the head turn to the patient's left or to the patient's right?","format":"lr"},
    ]),

    ("blank_stare", [
        {"id":"blank_stare_Q1","text":"Are the eyes open?","format":"yn"},
        {"id":"blank_stare_Q2","text":"Do the pupils of the eyes remain in a fixed position?","format":"yn"},
    ]),

    ("close_eyes", [
        {"id":"close_eyes_Q1","text":"Does the closure appear forceful, as if being squeezed shut?","format":"yn"},
    ]),

    ("eye_blinking", [
        {"id":"eye_blinking_Q1","text":"Is the blinking faster than 1Hz or more?","format":"yn"},
        {"id":"eye_blinking_Q2","text":"Does the blinking happen for at least a few seconds?","format":"yn"},
    ]),

    ("face_pulling", [
        {"id":"face_pulling_Q1","text":"Does the patient exhibit unilateral sustained face-pulling movements?","format":"yn"},
        {"id":"face_pulling_Q2","text":"Does the face pull to the patient’s left or to the patient’s right?","format":"lr"},
        {"id":"face_pulling_Q3","text":"Does the face pulling sustain for at least a few seconds?","format":"yn"},
    ]),

    ("face_twitching", [
        {"id":"face_twitching_Q1","text":"Are the twitches brief and repetitive?","format":"yn"},
        {"id":"face_twitching_Q2","text":"Does the face twitch to the patient’s left or to the patient’s right? (L/R)","format":"lr"},
    ]),

    ("tonic", [
        {"id":"tonic_Q1","text":"Does the patient exhibit sudden muscle stiffness or rigidity?","format":"yn"},
        {"id":"tonic_Q2","text":"Which body parts are affected?","format":"choice_multi","options_parts": ALLOWED_PARTS},
        {"id":"tonic_Q3","text":"For each affected body part, specify which side of the patient is involved? Select all that apply (sides or whole)","format":"choice_multi","options_sides": ALLOWED_SIDES},
    ]),

    ("clonic", [
        {"id":"clonic_Q1","text":"Are the movements rhythmic, stereotyped and repetitive jerks?","format":"yn"},
        {"id":"clonic_Q2","text":"Which body parts are affected?","format":"choice_multi","options_parts": ALLOWED_PARTS},
        {"id":"clonic_Q3","text":"For each affected body part, specify which side of the patient is involved? Select all that apply (sides or whole).","format":"choice_multi","options_sides": ALLOWED_SIDES},
    ]),

    ("arm_flexion", [
        {"id":"arm_flexion_Q1","text":"Are arms held in a flexed posture?","format":"yn"},
        {"id":"arm_flexion_Q2","text":"How many arms are held in a flexed posture?","format":"choice","options":["one","two"]},
        {"id":"arm_flexion_Q3","text":"Is this posture sustained?","format":"yn"},
        {"id":"arm_flexion_Q4","text":"For each affected arm, specify which side of the patient is involved?","format":"choice","options":["left arm","right arm","both arms"]},
    ]),

    ("arm_straightening", [
        {"id":"arm_straightening_Q1","text":"Are arms held in an extended posture?","format":"yn"},
        {"id":"arm_straightening_Q2","text":"How many arms are held in an extended posture?","format":"choice","options":["one","two"]},
        {"id":"arm_straightening_Q3","text":"Is this posture sustained?","format":"yn"},
        {"id":"arm_straightening_Q4","text":"For each affected arm, specify which side of the patient is involved?","format":"choice","options":["left arm","right arm","both arms"]},
    ]),

    ("figure4", [
        {"id":"figure4_Q1","text":"Is one arm extended while the other is flexed?","format":"yn"},
        {"id":"figure4_Q2","text":"Which arm of the patient is extended and which arm is flexed?","format":"choice","options":["left arm extended right arm flexed","right arm extended left arm flexed"]},
        {"id":"figure4_Q3","text":"Is this posture sustained?","format":"yn"},
    ]),

    ("oral_automatisms", [
        {"id":"oral_automatisms_Q1","text":"Does the patient have repetitive, stereotyped mouth or lip movements?","format":"yn"},
    ]),

    ("limb_automatisms", [
        {"id":"limb_automatisms_Q1","text":"Does the patient have repetitive stereotyped movements with their hands or legs?","format":"yn"},
        {"id":"limb_automatisms_Q2","text":"For each involved limb, specify which side of the patient is involved?","format":"choice_multi","options":["left arm","right arm","both arms","left leg","right leg","both legs"]},
    ]),

    ("pelvic_thrusting", [
        {"id":"pelvic_thrusting_Q1","text":"Are there repetitive anterior posterior movements of the hips?","format":"yn"},
    ]),

    ("full_body_shaking", [
        {"id":"full_body_shaking_Q1","text":"Does the patient experience shaking of the entire body, including arms, legs, and torso?","format":"yn"},
    ]),

    ("asynchronous_movement", [
        {"id":"asynchronous_movement_Q1","text":"Are the limbs of the patient shaking?","format":"yn"},
        {"id":"asynchronous_movement_Q2","text":"Which limbs of the patient are shaking? (list parts)","format":"choice_multi","options":["left arm","right arm","arms","left leg","right leg","legs","full body"]},
        {"id":"asynchronous_movement_Q3","text":"Are the limbs shaking at variable frequency, amplitude or both?","format":"choice","options":["frequency","amplitude","both","neither"]},
        {"id":"asynchronous_movement_Q4","text":"Do the movements appear non-stereotyped?","format":"yn"},
        {"id":"asynchronous_movement_Q5","text":"Are the limb movements asynchronous with respect to one another?","format":"yn"},
    ]),

    ("arms_move_simultaneously", [
        {"id":"arms_move_simultaneously_Q1","text":"Did the arms begin moving approximately at the same time?","format":"yn"},
    ]),

    ("verbal_responsiveness", [
        {"id":"verbal_responsiveness_Q1","text":"Did anyone address the patient verbally?","format":"yn"},
        {"id":"verbal_responsiveness_Q2","text":"If the patient was addressed verbally, did the patient provide a coherent answer?","format":"yn"},
    ]),

    ("ictal_vocalization", [
        {"id":"ictal_vocalization_Q1","text":"Is the patient making groaning, moaning, or guttural sounds?","format":"yn"},
        {"id":"ictal_vocalization_Q2","text":"Are the noises meaningless, stereotyped, and repetitive?","format":"yn"},
    ]),
])

# 权重
def weights_for(nq: int) -> List[float]:
    if nq == 1: return [1.0]
    if nq == 2: return [0.7, 0.3]
    if nq == 3: return [0.7, 0.15, 0.15]
    if nq == 4: return [0.7, 0.1,  0.1,  0.1]
    if nq == 5: return [0.7, 0.075, 0.075, 0.075, 0.075]
    return [1.0 / max(1, nq)] * max(1, nq)

# -----------------------------
# formalization
# -----------------------------
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

def norm_yn_strict(x: str) -> Optional[str]:
    s = norm_text(x)
    return s if s in {"yes", "no"} else None

def norm_lr(x: str) -> str:
    s = norm_text(x)
    if s in {"left","right"}: return s
    if s in {"l","r"}: return "left" if s=="l" else "right"
    return "neither"

def norm_choice(x: str, options: List[str]) -> str:
    s = norm_text(x)
    opts = set(o.lower() for o in options + ["unknown"])
    return s if s in opts else "unknown"

def norm_choice_multi(x: str, allowed: List[str]) -> str:
    s = _clean_raw(x).lower().strip()
    if not s or s == "unknown": return "unknown"
    toks = [t.strip() for t in re.split(r"[,\n;]+", s) if t.strip()]
    allow = [a.lower() for a in allowed]
    allow_set = set(allow)
    kept = [t for t in toks if t in allow_set]
    if not kept: return "unknown"
    order = {a:i for i,a in enumerate(allow)}
    kept = sorted(list(dict.fromkeys(kept)), key=lambda z: order.get(z, 10**9))
    return ", ".join(kept)

def norm_eq(a: str, b: str) -> bool:
    def clean(s: str) -> str:
        if s is None: return "unknown"
        s = str(s).strip().lower()
        s = (s.replace("，", ",").replace("；", ";").replace("|", ":"))
        s = " : ".join([t.strip() for t in s.split(":")]) if ":" in s else s
        return s
    a, b = clean(a), clean(b)
    if a == b: return True

    def as_set(s: str, sep: str):
        parts = [t.strip() for t in s.split(sep) if t.strip()]
        return set(parts)
    if any(ch in a for ch in ",;") or any(ch in b for ch in ",;"):
        for sep in [",", ";"]:
            A = as_set(a.replace(";", sep), sep)
            B = as_set(b.replace(";", sep), sep)
            if A and B and A == B:
                return True
    return False

# -----------------------------
# Prompt
# -----------------------------
SYSTEM_PROMPT = (
    "Return ONLY the JSON object requested. No explanations. "
    "Use lowercase only. For left/right questions use 'left'/'right'/'neither'. "
    "For choice: pick one of the provided options or 'unknown'. "
    "For choice_multi: return a comma-separated subset of the provided options, or 'unknown'. "
    "Never invent information not supported by the given text section."
)

def _feature_block(tag: str, questions: List[Dict[str, Any]], text: str) -> str:
    qlines, schema = [], []
    for q in questions:
        qid, fmt = q["id"], q["format"]
        if fmt == "yn":
            qlines.append(f'- {qid} (yn): {q["text"]}')
            schema.append(f'  "{qid}": "yes|no"')
        elif fmt == "lr":
            qlines.append(f'- {qid} (lr): {q["text"]}')
            schema.append(f'  "{qid}": "left|right|neither"')
        elif fmt == "choice":
            opts = [o.lower() for o in (q.get("options") or [])]
            qlines.append(f'- {qid} (choice): {q["text"]}  Options: {", ".join(opts)}')
            schema.append(f'  "{qid}": ONE OF [{", ".join(opts + ["unknown"])}]')
        elif fmt == "choice_multi":
            opts = [o.lower() for o in (q.get("options") or q.get("options_parts") or q.get("options_sides") or [])]
            qlines.append(f'- {qid} (choice_multi): {q["text"]}  Options: {", ".join(opts)} (comma-separated; or "unknown")')
            schema.append(f'  "{qid}": COMMA-LIST of subset of [{", ".join(opts)}] OR "unknown"')
        else:
            qlines.append(f'- {qid} (text): {q["text"]}')
            schema.append(f'  "{qid}": "unknown"')

    return (
f"""Section for feature: {tag}
Text:
{text}

Questions:
{chr(10).join(qlines)}
Schema keys for this section:
{chr(10).join(schema)}
""")

def _build_joint_prompt(items: List[Dict[str, str]]) -> str:
    """
    items: [{ "feature": str, "text": str }]
    把每个 feature 的文本与问题放成一个 Section，合并在一次调用里。
    """
    blocks, all_schema = [], []
    for it in items:
        feat = it["feature"]
        text = it["text"]
        qs = FEATURE_QUESTIONS.get(feat, [])
        blocks.append(_feature_block(feat, qs, text))
        for q in qs:
            fmt = q["format"]
            if fmt == "yn":
                all_schema.append(f'  "{q["id"]}": "yes|no"')
            elif fmt == "lr":
                all_schema.append(f'  "{q["id"]}": "left|right|neither"')
            elif fmt == "choice":
                opts = [o.lower() for o in (q.get("options") or [])]
                all_schema.append(f'  "{q["id"]}": ONE OF [{", ".join(opts + ["unknown"])}]')
            elif fmt == "choice_multi":
                opts = [o.lower() for o in (q.get("options") or q.get("options_parts") or q.get("options_sides") or [])]
                all_schema.append(f'  "{q["id"]}": COMMA-LIST of subset of [{", ".join(opts)}] OR "unknown"')
            else:
                all_schema.append(f'  "{q["id"]}": "unknown"')

    return (
f"""You will see multiple independent sections, each with its own text and a set of questions for a specific feature.
Answer ALL questions for ALL sections, and return ONE SINGLE JSON object whose keys are the question IDs across ALL sections.

Sections:
{chr(10).join(blocks)}

Output JSON (print ONLY this JSON, no backticks, no extra text):
{{
{",\n".join(all_schema)}
}}
""")

# -----------------------------
# 调一次 API
# -----------------------------
async def _ask_many_features_once(client: AsyncOpenAI, model: str, items: List[Dict[str, str]]) -> Dict[str, str]:
    """
    items: [{ "feature": <name>, "text": <justification text> }, ...]
    return: {qid -> normalized_answer_string}
    """
    if not items:
        return {}

    prompt = _build_joint_prompt(items)
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}],
        temperature=0,
    )
    txt = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(txt)
    except Exception:
        txt2 = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt.strip(), flags=re.I|re.M)
        data = json.loads(txt2)

    out: Dict[str, str] = {}
    for it in items:
        feat = it["feature"]
        for q in FEATURE_QUESTIONS.get(feat, []):
            qid = q["id"]; fmt = q["format"]
            raw = str(data.get(qid, "")).strip()
            if fmt == "yn":
                val = norm_text(raw)
                out[qid] = val if val in {"yes","no"} else val
            elif fmt == "lr":
                out[qid] = norm_lr(raw)
            elif fmt == "choice":
                opts = [o.lower() for o in (q.get("options") or [])]
                out[qid] = norm_choice(raw, opts)
            elif fmt == "choice_multi":
                opts = [o.lower() for o in (q.get("options") or q.get("options_parts") or q.get("options_sides") or [])]
                out[qid] = norm_choice_multi(raw, opts)
            else:
                out[qid] = norm_text(raw)
    return out

# -----------------------------
# 评分（feature yes/no gate -> Q1 gate -> 加权逐题）
# -----------------------------
def _score_one_feature(
    feature: str,
    gt_yesno: str,
    vlm_yesno: str,
    gt_ans: Dict[str, str],
    vlm_ans: Dict[str, str],
) -> float:
    # 1) gate1: feature 的 yes/no 
    g = norm_yn_strict(gt_yesno)
    v = norm_yn_strict(vlm_yesno)
    if (g is not None) and (v is not None) and (g != v):
        return 0.0

    # 2) gate2: Q1（若两边都为严格 yes/no 且不一致 -> 0）
    qids = [q["id"] for q in FEATURE_QUESTIONS.get(feature, [])]
    q1 = next((q for q in qids if q.endswith("_Q1")), None)
    if q1 is not None:
        g1 = norm_yn_strict(gt_ans.get(q1))
        v1 = norm_yn_strict(vlm_ans.get(q1))
        if (g1 is not None) and (v1 is not None) and (g1 != v1):
            return 0.0

    # 3) then using defined weights
    w = weights_for(len(qids))
    s = 0.0
    for j, qid in enumerate(qids):
        a = vlm_ans.get(qid)
        b = gt_ans.get(qid)
        if a is not None and b is not None and norm_eq(a, b):
            s += w[j]
    return max(0.0, min(1.0, s))

# -----------------------------
# 对外主函数（异步 + 同步封装）
# -----------------------------
async def score_video_features_async(
    client: AsyncOpenAI,
    model: str,
    video_name: str,
    items: List[Dict[str, str]],
) -> float:
    """
    计算一个视频在若干 feature 上的平均分
    items: list of dict，形如：
      {
        "feature": "close_eyes",
        "gt_yesno": "yes" / "no",
        "gt_text": "<ground truth justification text>",
        "vlm_yesno": "yes" / "no",
        "vlm_text": "<vlm justification text>"
      }
    返回：float，所有 feature 得分的平均值（若 items 空则返回 NaN）
    """
    valid_items = [it for it in items if it.get("feature") in FEATURE_QUESTIONS]
    if not valid_items:
        return float("nan")

    # 1) 各自批量问一次：GT & VLM
    gt_query = [{"feature": it["feature"], "text": it.get("gt_text", "")} for it in valid_items]
    vlm_query = [{"feature": it["feature"], "text": it.get("vlm_text", "")} for it in valid_items]

    gt_answers = await _ask_many_features_once(client, model, gt_query)
    vlm_answers = await _ask_many_features_once(client, model, vlm_query)

    # 2) feature 打分
    scores = []
    for it in valid_items:
        f = it["feature"]
        score_f = _score_one_feature(
            feature=f,
            gt_yesno=it.get("gt_yesno", ""),
            vlm_yesno=it.get("vlm_yesno", ""),
            gt_ans=gt_answers,
            vlm_ans=vlm_answers,
        )
        scores.append(score_f)

    # 3) 平均
    if not scores:
        return float("nan")
    avg = sum(scores) / len(scores)
    # two decimal
    return float(Decimal(str(avg)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def score_video_features(
    api_key: str,
    model: str,
    video_name: str,
    items: List[Dict[str, str]],
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
) -> float:
    """
    同步封装：内部创建 AsyncOpenAI 客户端并执行异步函数
    """
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return asyncio.run(score_video_features_async(client, model, video_name, items))
