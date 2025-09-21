# Download the bertscore model (run in terminal; replace <HF_CACHE_DIR> with your path):
# huggingface-cli download microsoft/deberta-xlarge-mnli --cache-dir <HF_CACHE_DIR>

import os, re
import numpy as np
import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# ================== Environment ==================
# Prefer environment variable; fallback to a placeholder (replace <HF_CACHE_DIR>).
HF_CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE", "<HF_CACHE_DIR>")
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR

# ================== Batch Config ==================
# Base directory for your project; set via env PROJECT_BASE_DIR or replace <BASE_DIR>.
BASE_DIR = os.environ.get("PROJECT_BASE_DIR", "<BASE_DIR>")

# Models to evaluate (file names will be formatted with {model}), AF3 should run another code Task2_AF3_metrics.py
MODELS = [    
    "InternVL3_5-8B",
    "InternVL3_5-38B",
    "Qwen2.5-VL-7B-Instruct",
    "Qwen2.5-VL-32B-Instruct",
    "Qwen2.5-VL-72B-Instruct",
    "Lingshu-32B",
    "Qwen2.5-Omni-7B",
]

# Pattern of prediction CSVs (must contain "{model}")
# Example relative pattern: task2_metrics/Task1_InternVL3_5-8B_all_merged_llmmerge.csv
PRED_PATTERN = os.path.join(BASE_DIR, "task2_metrics", "Task1_{model}_all_merged_llmmerge.csv")

# Ground-truth CSV (shared by all models)
GT_CSV = os.path.join(BASE_DIR, "task2_metrics", "task12_annotation.csv")

# Output root directory (each model will have its own subfolder)
OUT_ROOT = os.path.join(BASE_DIR, "task2_metrics", "Task2_llmmerge_metrics_results")

# ================== Evaluation Settings ==================
ID_COL = "file_name"
LOWERCASE = True
LANG = "en"  # language for BERTScore
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"
RESCALE_BERT = False
FAIL_REGEX = re.compile(r"^\s*(fail(ed)?|error|n/?a|none|null)?\s*$", re.I)

# ================== Helpers ==================
def feat_name(col: str) -> str:
    """Strip the 'justification_for_' prefix to get the feature name."""
    return col[len("justification_for_"):]

def to_long(df: pd.DataFrame, id_col: str, feat_cols: list[str], label: str) -> pd.DataFrame:
    """Pivot wide justification columns into long format: [id, label, feature]."""
    parts = []
    for c in feat_cols:
        tmp = df[[id_col, c]].copy()
        tmp.columns = [id_col, label]
        tmp["feature"] = feat_name(c)
        parts.append(tmp)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[id_col, label, "feature"])

def is_fail(s) -> bool:
    """Return True for blank/NA/fail-like strings."""
    if s is None:
        return True
    if isinstance(s, float) and np.isnan(s):
        return True
    s = str(s)
    return bool(FAIL_REGEX.match(s)) or (s.strip() == "")

def corpus_bleu(preds, refs):
    """SacreBLEU corpus score with safe fallback."""
    try:
        return sacrebleu.corpus_bleu(preds, [refs]).score
    except Exception:
        return float("nan")

def evaluate_one_model(model_name: str) -> pd.DataFrame:
    """Evaluate a single model and return its per-feature summary (with an AVG row)."""
    pred_csv = PRED_PATTERN.format(model=model_name)
    out_dir = os.path.join(OUT_ROOT, model_name)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(pred_csv):
        print(f"[WARN] Prediction CSV not found for {model_name}: {pred_csv}")
        # Produce an empty file to keep structure consistent
        empty = pd.DataFrame(columns=[
            "feature","n_pairs","bleu_corpus","rouge1_f1_mean","rougeL_f1_mean","berts_f1_mean"
        ])
        empty.to_csv(os.path.join(out_dir, f"Task1_{model_name}_llmmerge_feature_summary.csv"), index=False)
        return pd.DataFrame()  # return truly empty to skip in overall concat

    if not os.path.exists(GT_CSV):
        raise FileNotFoundError(f"Ground truth CSV not found: {GT_CSV}")

    pred = pd.read_csv(pred_csv)
    gt = pd.read_csv(GT_CSV)

    # Discover justification columns
    J_PREFIX = "justification_for_"
    feat_cols = [c for c in pred.columns if c.startswith(J_PREFIX)]
    if not feat_cols:
        print(f"[WARN] {model_name}: No columns start with '{J_PREFIX}' in prediction CSV.")
        empty = pd.DataFrame(columns=[
            "feature","n_pairs","bleu_corpus","rouge1_f1_mean","rougeL_f1_mean","berts_f1_mean"
        ])
        empty.to_csv(os.path.join(out_dir, f"Task1_{model_name}_llmmerge_feature_summary.csv"), index=False)
        return pd.DataFrame()

    # Wide -> long
    pred_long = to_long(pred, ID_COL, feat_cols, "pred_text")
    gt_long   = to_long(gt,   ID_COL, feat_cols, "gt_text")

    # Left-join on [id, feature]
    merged = pred_long.merge(gt_long, on=[ID_COL, "feature"], how="left", validate="m:1")

    # Filter invalid predictions; also drop rows with empty GT
    merged = merged[~merged["pred_text"].apply(is_fail)].copy()
    merged["gt_text"] = merged["gt_text"].fillna("").astype(str)
    merged = merged[merged["gt_text"].str.strip() != ""].copy()

    # Normalize case/whitespace if requested
    if LOWERCASE:
        merged["pred_eval"] = merged["pred_text"].astype(str).str.strip().str.lower()
        merged["gt_eval"]   = merged["gt_text"].astype(str).str.strip().str.lower()
    else:
        merged["pred_eval"] = merged["pred_text"].astype(str).str.strip()
        merged["gt_eval"]   = merged["gt_text"].astype(str).str.strip()

    if merged.empty:
        print(f"[INFO] {model_name}: Nothing to evaluate after filtering.")
        empty = pd.DataFrame(columns=[
            "feature","n_pairs","bleu_corpus","rouge1_f1_mean","rougeL_f1_mean","berts_f1_mean"
        ])
        empty.to_csv(os.path.join(out_dir, f"Task1_{model_name}_llmmerge_feature_summary.csv"), index=False)
        return pd.DataFrame()

    # Metric calculators
    rscorer = rouge_scorer.RougeScorer(["rouge1","rougeL"], use_stemmer=True)

    # Aggregate per feature
    rows = []
    for f, grp in merged.groupby("feature"):
        preds = grp["pred_eval"].tolist()
        refs  = grp["gt_eval"].tolist()
        n = len(grp)

        # ROUGE (F1) mean across pairs
        r1, rL = [], []
        for c, r in zip(preds, refs):
            sc = rscorer.score(r, c)
            r1.append(sc["rouge1"].fmeasure)
            rL.append(sc["rougeL"].fmeasure)

        # BERTScore (F1) mean across pairs
        _, _, F1 = bert_score(
            preds, refs, lang=LANG,
            model_type=BERTSCORE_MODEL,
            rescale_with_baseline=RESCALE_BERT
        )

        rows.append({
            "feature": f,
            "n_pairs": n,
            "bleu_corpus": corpus_bleu(preds, refs),
            "rouge1_f1_mean": float(np.mean(r1)),
            "rougeL_f1_mean": float(np.mean(rL)),
            "berts_f1_mean": float(F1.numpy().mean()),
        })

    feat_sum = pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)

    # Append an AVG row (macro mean across features)
    numeric_cols = ["n_pairs", "bleu_corpus", "rouge1_f1_mean", "rougeL_f1_mean", "berts_f1_mean"]
    avg_row = {"feature": "AVG"}
    avg_row["n_pairs"] = int(round(feat_sum["n_pairs"].mean()))  # change to sum() if desired
    for col in numeric_cols[1:]:
        avg_row[col] = float(feat_sum[col].mean())

    feat_sum = pd.concat([feat_sum, pd.DataFrame([avg_row])], ignore_index=True)

    # Save per-model summary
    out_path = os.path.join(out_dir, f"Task1_{model_name}_llmmerge_feature_summary.csv")
    feat_sum.to_csv(out_path, index=False)
    print(f"[OK] {model_name} -> {out_path}")

    # Add model column for the overall summary
    feat_sum.insert(0, "model", model_name)
    return feat_sum


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    # Evaluate each model in order
    all_summaries = []
    for m in MODELS:
        print(f"==== Evaluating: {m} ====")
        try:
            df = evaluate_one_model(m)
            if not df.empty:
                all_summaries.append(df)
        except Exception as e:
            print(f"[ERROR] {m}: {e}")

    # Concatenate all model summaries (optional overall file)
    if all_summaries:
        overall = pd.concat(all_summaries, ignore_index=True)
        overall_path = os.path.join(OUT_ROOT, "overall_summary.csv")
        overall.to_csv(overall_path, index=False)
        print(f"[OK] Overall summary -> {overall_path}")
    else:
        print("[INFO] No summaries produced.")


if __name__ == "__main__":
    main()

