
import pandas as pd
from rapidfuzz.distance import Levenshtein, LCSseq
from sklearn.metrics import f1_score
import re
import argparse
import os

def tokenize(seq_str: str) -> list[str]:
    if pd.isna(seq_str):
        return []
    parts = [p for p in re.split(r",", str(seq_str))]
    tokens = []
    for p in parts:
        t = p.strip().strip('"').strip("'").strip()
        t = t.rstrip(".;")
        t = re.sub(r"\s+", "_", t)
        if t:
            tokens.append(t)
    return tokens

def align_for_f1(pred_tokens: list[str], gt_tokens: list[str]):
    y_true = []
    y_pred = []
    for op in Levenshtein.opcodes(pred_tokens, gt_tokens):
        tag = op.tag
        s1, e1, s2, e2 = op.src_start, op.src_end, op.dest_start, op.dest_end
        if tag == "equal":
            y_pred.extend(pred_tokens[s1:e1])
            y_true.extend(gt_tokens[s2:e2])
        elif tag == "replace":
            for i in range(e1 - s1):
                y_pred.append(pred_tokens[s1 + i])
                y_true.append(gt_tokens[s2 + i])
        elif tag == "delete":
            for i in range(e1 - s1):
                y_pred.append("<GAP>")
                y_true.append(gt_tokens[s2] if s2 < len(gt_tokens) else "<GAP>")
        elif tag == "insert":
            for i in range(e2 - s2):
                y_pred.append(pred_tokens[s1] if s1 < len(pred_tokens) else "<GAP>")
                y_true.append(gt_tokens[s2 + i])
    n = min(len(y_true), len(y_pred))
    return y_true[:n], y_pred[:n]

def main(pred_path: str, gt_path: str, out_path: str):
    # pred_df = pd.read_csv(pred_path)
    # gt_df = pd.read_csv(gt_path)

    gt_df = pd.read_csv(gt_path, encoding="utf-8")
    #print(gt_df.head(5))
    pred_df = pd.read_csv(pred_path, encoding="utf-8")
    #print(pred_df.head(5))

    df = pred_df.merge(gt_df, on="file_name", how="inner")

    records = []
    for _, row in df.iterrows():
        fname = row["file_name"]
        pred_tokens = tokenize(row["feature_list"])
        gt_tokens = tokenize(row["event_sequence"])
        edit_dist = Levenshtein.distance(pred_tokens, gt_tokens)
        lcs_len = LCSseq.similarity(pred_tokens, gt_tokens)
        lcs_ratio_gt = (lcs_len / len(pred_tokens)) if len(pred_tokens) > 0 else 0.0

        if len(pred_tokens) == 0 and len(gt_tokens) == 0:
            temporal_f1 = 1.0
        else:
            y_true, y_pred = align_for_f1(pred_tokens, gt_tokens)
            try:
                temporal_f1 = float(f1_score(y_true, y_pred, average="micro"))
            except Exception:
                temporal_f1 = 0.0

        records.append({
            "file_name": fname,
            "pred_tokens":pred_tokens,
            "gt_tokens":gt_tokens,
            "edit_distance": edit_dist,
            "temporal_f1": temporal_f1,
            "lcs_ratio_gt": lcs_ratio_gt,
            "pred_len": len(pred_tokens),
            "gt_len": len(gt_tokens),
        })

    metrics_df = pd.DataFrame.from_records(records)
    metrics_df.to_csv(out_path, index=False)

    dataset_metrics = {
        #"num_pairs": int(metrics_df.shape[0]),
        # "edit_distance": float(metrics_df["edit_distance"].mean()) if not metrics_df.empty else 0.00,
        # "temporal_f1": float(metrics_df["temporal_f1"].mean()) if not metrics_df.empty else 0.00,
        # "lcs_ratio": float(metrics_df["lcs_ratio_gt"].mean()) if not metrics_df.empty else 0.00,

        "edit_distance": round(metrics_df["edit_distance"].mean(), 2) if not metrics_df.empty else 0.00,
        "temporal_f1":   round(metrics_df["temporal_f1"].mean(), 2)   if not metrics_df.empty else 0.00,
        "lcs_ratio":     round(metrics_df["lcs_ratio_gt"].mean(), 2)  if not metrics_df.empty else 0.00,

    }
    return dataset_metrics



if __name__ == "__main__":

    model_names = [
        'Qwen2.5-VL-7B-Instruct',
        'InternVL3_5-8B',
        'Qwen2.5-VL-32B-Instruct',
        'InternVL3_5-38B',
        'Qwen2.5-VL-72B-Instruct',
        'audio-flamingo-3',
        'Qwen2.5-Omni-7B',
        'Lingshu-32B',
        'Qwen3-VL-8B-Instruct',
        'Qwen3-VL-32B-Instruct',
    ]
    base_dir = '/home/lina/ssb/SeizureSemiologyBench/'
    metric_rows = []
    for model in model_names:
        pred_csv = f"{base_dir}result/vlm_inference/{model}/Task5_{model}_all_merge.csv"
        if os.path.isfile(pred_csv):
            gt_csv = f"{base_dir}result/ground_truth/task5_sequence_annotation_vlm.csv"
            if("omni" in model.lower()):
                gt_csv = f"{base_dir}result/ground_truth/task5_sequence_annotation.csv"
            out_csv = f"{base_dir}metrics/task5/task5_sequence_3_metrics_{model}.csv"
            model_metrics = main(pred_csv,gt_csv, out_csv)
            model_metrics["model"] = model
            metric_rows.append(model_metrics)

    out_df = pd.DataFrame(metric_rows, columns=["model","edit_distance", "temporal_f1","lcs_ratio"])
    out_path = f"{base_dir}metrics/task5_sequence_3_metrics.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8")





