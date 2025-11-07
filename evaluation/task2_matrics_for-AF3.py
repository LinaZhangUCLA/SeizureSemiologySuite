#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
from typing import List, Dict, Optional
from decimal import Decimal, ROUND_HALF_UP

def round_up_2(x: float) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

# ===== feature -> question_lists =====
FEATURE_QUESTIONS = {
    "occur_during_sleep": ["occur_during_sleep_Q1"],

    "head_turning": ["head_turning_Q1", "head_turning_Q2"],
    "blank_stare": ["blank_stare_Q1", "blank_stare_Q2"],
    "close_eyes": ["close_eyes_Q1"],
    "eye_blinking": ["eye_blinking_Q1", "eye_blinking_Q2"],
    "face_pulling": ["face_pulling_Q1", "face_pulling_Q2", "face_pulling_Q3"],
    "face_twitching": ["face_twitching_Q1", "face_twitching_Q2"],

    "tonic": ["tonic_Q1", "tonic_Q2", "tonic_Q3"],
    "clonic": ["clonic_Q1", "clonic_Q2", "clonic_Q3"],
    "arm_flexion": ["arm_flexion_Q1", "arm_flexion_Q2", "arm_flexion_Q3","arm_flexion_Q4"],
    "arm_straightening": ["arm_straightening_Q1", "arm_straightening_Q2", "arm_straightening_Q3","arm_straightening_Q4"],
    "figure4": ["figure4_Q1", "figure4_Q2", "figure4_Q3"],

    "oral_automatisms": ["oral_automatisms_Q1"],
    "limb_automatisms": ["limb_automatisms_Q1", "limb_automatisms_Q2"],
    "pelvic_thrusting": ["pelvic_thrusting_Q1"],
    "full_body_shaking": ["full_body_shaking_Q1"],

    "asynchronous_movement": ["asynchronous_movement_Q1", "asynchronous_movement_Q2", "asynchronous_movement_Q3", "asynchronous_movement_Q4","asynchronous_movement_Q5"],
    "arms_move_simultaneously": ["arms_move_simultaneously_Q1"],
    "verbal_responsiveness": ["verbal_responsiveness_Q1", "verbal_responsiveness_Q2"],
    "ictal_vocalization": ["ictal_vocalization_Q1", "ictal_vocalization_Q2"],
}

# weights
def weights_for(nq: int):
    if nq == 1: return [1.0]
    if nq == 2: return [0.7, 0.3]
    if nq == 3: return [0.7, 0.15, 0.15]
    if nq == 4: return [0.7, 0.1,  0.1,  0.1]
    if nq == 5: return [0.7, 0.075, 0.075, 0.075, 0.075]
    return [1.0 / max(1, nq)] * max(1, nq)

def norm_eq(a: str, b: str, qid: str) -> bool:
    def clean(s: str) -> str:
        if s is None: return "unknown"
        s = str(s).strip().lower()
        s = (s
             .replace("，", ",")
             .replace("；", ";")
             .replace("|", ":"))
        s = " : ".join([t.strip() for t in s.split(":")]) if ":" in s else s
        return s

    a, b = clean(a), clean(b)
    if a == b:
        return True

    def as_set(s: str, sep: str):
        parts = [t.strip() for t in s.split(sep) if t.strip()]
        return set(parts)

    if (("," in a) or (";" in a)) or (("," in b) or (";" in b)):
        for sep in [",", ";"]:
            A = as_set(a.replace(";", sep), sep)
            B = as_set(b.replace(";", sep), sep)
            if A and B and A == B:
                return True

    return False

def _yn_strict(x: str) -> Optional[str]:
    s = ("" if x is None else str(x)).strip().lower()
    return s if s in {"yes", "no"} else None

def build_present_feature_map(df_gt: pd.DataFrame, df_v: pd.DataFrame) -> Dict[str, List[str]]:
    gt_name_col = df_gt.columns[0]
    v_name_col  = df_v.columns[0]
    gt_cols = set(df_gt.columns) - {gt_name_col}
    v_cols  = set(df_v.columns)  - {v_name_col}

    feat_qs_map: Dict[str, List[str]] = {}
    for feat, qids in FEATURE_QUESTIONS.items():
        present = [q for q in qids if (q in gt_cols and q in v_cols)]
        if present:
            feat_qs_map[feat] = present
    return feat_qs_map

# ===  ===
def _name_stem(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    s = os.path.basename(s)
    s = s.lower()
    s = os.path.splitext(s)[0]
    return s

def align_and_fill(df_gt: pd.DataFrame, df_v: pd.DataFrame):
    gt_name_col = df_gt.columns[0]
    v_name_col  = df_v.columns[0]
    df_gt = df_gt.copy()
    df_v  = df_v.copy()
    df_gt.rename(columns={gt_name_col: "file_name"}, inplace=True)
    df_v.rename(columns={v_name_col: "file_name"}, inplace=True)

    # make no differences between.mp4/.wav 
    df_gt["__key__"] = df_gt["file_name"].map(_name_stem)
    df_v["__key__"]  = df_v["file_name"].map(_name_stem)

    df_gt = df_gt.drop_duplicates(subset=["__key__"], keep="first")
    df_v  = df_v.drop_duplicates(subset=["__key__"],  keep="first")

    common_keys = sorted(set(df_gt["__key__"]).intersection(set(df_v["__key__"])))
    df_gt = df_gt[df_gt["__key__"].isin(common_keys)].reset_index(drop=True)

    # reorder the vlm as the order of GT
    order = {k: i for i, k in enumerate(df_gt["__key__"].tolist())}
    df_v = df_v[df_v["__key__"].isin(common_keys)].copy()
    df_v["__ord__"] = df_v["__key__"].map(order)
    df_v.sort_values("__ord__", inplace=True)
    df_v.drop(columns="__ord__", inplace=True)
    df_v.reset_index(drop=True, inplace=True)

    # using intersection
    feat_qs_map = build_present_feature_map(df_gt, df_v)
    keep_qids = [q for qids in feat_qs_map.values() for q in qids]

  
    keep_feats = [feat for feat in FEATURE_QUESTIONS.keys()
                  if (feat in df_gt.columns and feat in df_v.columns)]

    keep_cols = ["file_name"] + keep_feats + keep_qids
    df_gt = df_gt[[c for c in keep_cols if c in df_gt.columns]]
    df_v  = df_v [[c for c in keep_cols if c in df_v.columns]]

    return df_gt, df_v, feat_qs_map, keep_feats

def score_matrix(df_gt: pd.DataFrame, df_v: pd.DataFrame,
                 feat_qs_map: Dict[str, List[str]],
                 keep_feats: List[str]):
    rows = []
    feats_order = [f for f in FEATURE_QUESTIONS.keys() if f in feat_qs_map]

    for i in range(len(df_gt)):
        row_scores = {"file_name": df_gt.loc[i, "file_name"]}
        for feat in feats_order:
            qids = feat_qs_map[feat]
            if not qids:
                continue

            # 1) compare feature's yes/no
            feat_gate_available = (feat in df_gt.columns) and (feat in df_v.columns)
            if feat_gate_available:
                gt_feat = _yn_strict(df_gt.loc[i, feat])
                v_feat  = _yn_strict(df_v.loc[i, feat])
                if (gt_feat is not None) and (v_feat is not None) and (gt_feat != v_feat):
                    row_scores[feat] = 0.0
                    continue

            # 2)  Q1  yes/no
            q1 = next((q for q in qids if q.endswith("_Q1")), None)
            if q1 is not None:
                gt_q1 = _yn_strict(df_gt.loc[i, q1]) if q1 in df_gt.columns else None
                v_q1  = _yn_strict(df_v.loc[i,  q1]) if q1 in df_v.columns else None
                if (gt_q1 is not None) and (v_q1 is not None) and (gt_q1 != v_q1):
                    row_scores[feat] = 0.0
                    continue

            # 3) grading as weighted scores
            w = weights_for(len(qids))
            s = 0.0
            for j, qid in enumerate(qids):
                a = df_v.loc[i, qid] if qid in df_v.columns else None
                b = df_gt.loc[i, qid] if qid in df_gt.columns else None
                if norm_eq(a, b, qid):
                    s += w[j]
            row_scores[feat] = max(0.0, min(1.0, s))
        rows.append(row_scores)
    return pd.DataFrame(rows)

def make_feature_mean_row(score_df: pd.DataFrame, model_name: str,
                          feats_present: List[str], keep_filename_col=False):
    means = score_df[feats_present].mean(axis=0).to_dict()
    row1 = {"": ""}           
    row2 = {"": model_name}  
    for f in feats_present:
        row1[f] = f
        row2[f] = round_up_2(float(means[f]))
    out = pd.DataFrame([row1, row2])
    if keep_filename_col:
        pass
    return out

def make_overall_score_csv(score_df: pd.DataFrame, model_name: str,
                           feats_present: List[str]):
    if len(feats_present) == 0:
        overall = float("nan")
    else:
        per_feature_means = score_df[feats_present].mean(axis=0)
        overall = float(per_feature_means.mean())
    out = pd.DataFrame([
        {"": "", "question_list_score": "question_list_score"},
        {"": model_name, "question_list_score": (round_up_2(overall) if overall == overall else "")}
    ])
    return out

def parse_args():
    ap = argparse.ArgumentParser("Score VLM vs Ground Truth with feature yes/no gating + Q1 gating + weighted questions (ignore file extension)")
    ap.add_argument("--gt-csv", required=True, help="Ground truth CSV（已合并 feature 的 yes/no 到同一表）")
    ap.add_argument("--vlm-csv", required=True, help="VLM CSV（已合并 feature 的 yes/no 到同一表）")
    ap.add_argument("--model-name", required=True, help="模型名称（会写进 2×(M+1) 和 2×2 的第二行第一列）")
    ap.add_argument("--out-dir", required=True, help="输出目录")
    ap.add_argument("--drop-file-name-in-matrix", action="store_true",
                    help="若设定，则第一张矩阵不包含 file_name 列（得到严格 N×M）")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df_gt = pd.read_csv(args.gt_csv, dtype=str)
    df_v  = pd.read_csv(args.vlm_csv, dtype=str)

    df_gt, df_v, feat_qs_map, keep_feats = align_and_fill(df_gt, df_v)

    feats_present = [f for f in FEATURE_QUESTIONS.keys() if f in feat_qs_map and len(feat_qs_map[f]) > 0]

    score_df = score_matrix(df_gt, df_v, feat_qs_map, keep_feats)

    # Table1：video×feature
    matrix_df = score_df.drop(columns=["file_name"]) if args.drop_file_name_in_matrix else score_df
    p1 = os.path.join(args.out_dir, f"{args.model_name}__per_video_feature_scores.csv")
    matrix_df.to_csv(p1, index=False)

    # Table2：model×feature（rounded）
    per_feat_df = make_feature_mean_row(score_df, args.model_name, feats_present)
    p2 = os.path.join(args.out_dir, f"{args.model_name}__per_feature_mean_scores.csv")
    per_feat_df.to_csv(p2, index=False)

    # Table3：overall（rounded）
    overall_df = make_overall_score_csv(score_df, args.model_name, feats_present)
    p3 = os.path.join(args.out_dir, f"{args.model_name}__overall_score.csv")
    overall_df.to_csv(p3, index=False)

    print("Saved:")
    print(" -", p1)
    print(" -", p2)
    print(" -", p3)

if __name__ == "__main__":
    main()
