#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, argparse
import pandas as pd

MASTER_FEATURE_HEADER = [
    "model_name",
    "occur_during_sleep","head_turning","blank_stare","close_eyes","eye_blinking",
    "face_pulling","face_twitching","tonic","clonic",
    "arm_flexion","arm_straightening","figure4",
    "oral_automatisms","limb_automatisms","pelvic_thrusting","full_body_shaking",
    "asynchronous_movement","arms_move_simultaneously","verbal_responsiveness","ictal_vocalization",
]

def parse_args():
    ap = argparse.ArgumentParser("Aggregate existing per-model outputs into two master CSVs (no API calls).")
    ap.add_argument("--root", required=True, help="根目录：包含各模型子目录（每个子目录里已有三张表); you need to have the root folder including three existing csv for each of the model")
    ap.add_argument("--feature-out", default="__MASTER__per_feature_mean_scores.csv",
                    help="总表（特征级）文件名（默认 __MASTER__per_feature_mean_scores.csv）; feature-level csv, file name default as __MASTER__per_feature_mean_scores.csv")
    ap.add_argument("--overall-out", default="__MASTER__overall_scores.csv",
                    help="总表（模型级）文件名（默认 __MASTER__overall_scores.csv）; model-level csv, file name default as __MASTER__overall_scores.csv")
    return ap.parse_args()

def read_second_row_dict(csv_path: str) -> dict | None:
    try:
        df = pd.read_csv(csv_path, dtype=str)
        if df.shape[0] < 2:
            return None
        return df.iloc[1].to_dict()
    except Exception:
        return None

def main():
    args = parse_args()
    root = args.root
    out_feature = os.path.join(root, args.feature_out)
    out_overall  = os.path.join(root, args.overall_out)

    feature_rows = []
    overall_rows = []

    seen_models_feature = set()
    seen_models_overall = set()

    subdirs = sorted([p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)])
    for d in subdirs:
        base = os.path.basename(d)
        if base.startswith("__MASTER__"):
            continue

        pfms = sorted(glob.glob(os.path.join(d, "*__per_feature_mean_scores.csv")))
        ovos = sorted(glob.glob(os.path.join(d, "*__overall_score.csv")))

        if pfms:
            dct = read_second_row_dict(pfms[0])
            if dct is not None:
                model_name = (dct.get("") or base).strip()
                if model_name not in seen_models_feature:
                    row = {h: "" for h in MASTER_FEATURE_HEADER}
                    row["model_name"] = model_name
                    for feat in MASTER_FEATURE_HEADER[1:]:
                        if feat in dct and str(dct[feat]).strip() != "":
                            row[feat] = str(dct[feat]).strip()
                    feature_rows.append(row)
                    seen_models_feature.add(model_name)

        if ovos:
            dct2 = read_second_row_dict(ovos[0])
            if dct2 is not None:
                model_name2 = (dct2.get("") or base).strip()
                if model_name2 not in seen_models_overall:
                    score = str(dct2.get("question_list_score", "") or "").strip()
                    overall_rows.append({"model_name": model_name2, "question_list_score": score})
                    seen_models_overall.add(model_name2)

    df_feat = pd.DataFrame(feature_rows, columns=MASTER_FEATURE_HEADER)
    df_over = pd.DataFrame(overall_rows, columns=["model_name","question_list_score"])

    os.makedirs(root, exist_ok=True)
    df_feat.to_csv(out_feature, index=False)
    df_over.to_csv(out_overall, index=False)

    print(f"[OK] Wrote feature-level master -> {out_feature}  (rows={len(df_feat)})")
    print(f"[OK] Wrote overall master  -> {out_overall}   (rows={len(df_over)})")

if __name__ == "__main__":
    main()
