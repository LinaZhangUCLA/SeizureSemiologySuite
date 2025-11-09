#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import pandas as pd

# mode*feature names
MASTER_FEATURE_HEADER = [
    "model_name",
    "occur_during_sleep","head_turning","blank_stare","close_eyes","eye_blinking",
    "face_pulling","face_twitching","tonic","clonic",
    "arm_flexion","arm_straightening","figure4",
    "oral_automatisms","limb_automatisms","pelvic_thrusting","full_body_shaking",
    "asynchronous_movement","arms_move_simultaneously","verbal_responsiveness","ictal_vocalization",
]

def parse_args():
    ap = argparse.ArgumentParser("Aggregate flat model csvs into two master csvs.")
    ap.add_argument(
        "--root",
        #required=True,
        default="/home/lina/ssb/SeizureSemiologyBench/metrics_test/middle_results/",
        help="目录：里面直接是诸如 Model__per_feature_mean_scores.csv / Model__overall_score.csv 这种文件"
    )
    ap.add_argument(
        "--feature-out",
        default="Task2_feature_question_list_scores.csv",
        help="输出的特征级总表文件名"
    )
    ap.add_argument(
        "--overall-out",
        default="Task2_overall_question_list_scores.csv",
        help="输出的模型级总表文件名"
    )
    return ap.parse_args()

def read_second_row_dict(csv_path: str):
    """read the csv files，then return dict."""
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
    out_overall = os.path.join(root, args.overall_out)

    # root =  
    # out_feature = 
    # out_overall = 

    # finding file
    per_feature_files = glob.glob(os.path.join(root, "*__per_feature_mean_scores.csv"))
    overall_files     = glob.glob(os.path.join(root, "*__overall_score.csv"))

    feature_rows = []
    overall_rows = []
    seen_feature_models = set()
    seen_overall_models = set()
    print("find files: ",per_feature_files)
    # processing feature-level
    for f in sorted(per_feature_files):
        dct = read_second_row_dict(f)
        if not dct:
            continue
        #  get model_name
        model_name = (dct.get("") or os.path.basename(f).split("__", 1)[0]).strip()
        if model_name in seen_feature_models:
            continue
        row = {h: "" for h in MASTER_FEATURE_HEADER}
        row["model_name"] = model_name
        for feat in MASTER_FEATURE_HEADER[1:]:
            if feat in dct and str(dct[feat]).strip() != "":
                row[feat] = str(dct[feat]).strip()
        feature_rows.append(row)
        seen_feature_models.add(model_name)

    # processing overall score for each model
    for f in sorted(overall_files):
        dct = read_second_row_dict(f)
        if not dct:
            continue
        model_name = (dct.get("") or os.path.basename(f).split("__", 1)[0]).strip()
        if model_name in seen_overall_models:
            continue
        score = str(dct.get("question_list_score", "") or "").strip()
        overall_rows.append({"model_name": model_name, "question_list_score": score})
        seen_overall_models.add(model_name)

    # output
    df_feat = pd.DataFrame(feature_rows, columns=MASTER_FEATURE_HEADER)
    df_over = pd.DataFrame(overall_rows, columns=["model_name", "question_list_score"])

    os.makedirs(root, exist_ok=True)
    df_feat.to_csv(out_feature, index=False)
    df_over.to_csv(out_overall, index=False)

    print(f"[OK] Wrote feature-level master -> {out_feature} (rows={len(df_feat)})")
    print(f"[OK] Wrote overall master  -> {out_overall} (rows={len(df_over)})")

if __name__ == "__main__":
    main()
