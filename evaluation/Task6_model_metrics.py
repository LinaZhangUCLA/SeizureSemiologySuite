import os, re
import numpy as np
import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import torch

# ===================== Environment =====================
# Prefer env var to avoid hardcoding host paths; replace <HF_CACHE_DIR>/<BASE_DIR> if not using env.
os.environ["TRANSFORMERS_CACHE"] = os.environ.get("TRANSFORMERS_CACHE", "<HF_CACHE_DIR>")
BASE_DIR = os.environ.get("PROJECT_BASE_DIR", "<BASE_DIR>")

# ===================== Batch Config =====================
# Ground truth CSV (shared by all models)
GT_CSV = os.path.join(BASE_DIR, "task6_metrics", "task6_annotation_new.csv")

# Input/Output patterns (must contain {model})
# Example input:  <BASE_DIR>/task6_metrics/Task6_llmmerge/Task6_Qwen2.5-VL-72B-Instruct_all_merged_llmmerge.csv
# Example output: <BASE_DIR>/task6_metrics/Task6_llmmerge_results/Task6_Qwen2.5-VL-72B-Instruct_metrics_summary.csv
PRED_PATTERN = os.path.join(BASE_DIR, "task6_metrics", "Task6_llmmerge", "Task6_{model}_all_merged_llmmerge.csv")
OUT_DIR      = os.path.join(BASE_DIR, "task6_metrics", "Task6_llmmerge_results")

# Models to run (AF3 excluded)
MODELS = [
    "InternVL3_5-8B",
    "InternVL3_5-38B",
    "Qwen2.5-VL-7B-Instruct",
    "Qwen2.5-VL-32B-Instruct",
    "Qwen2.5-VL-72B-Instruct",
    "Lingshu-32B",
]

# Column names
ID_COL        = "file_name"
PRED_TEXT_COL = "report"
GT_TEXT_COL   = "report"

# ===================== Evaluation Settings =====================
LOWERCASE       = True
LANG            = "en"
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"
RESCALE_BERT    = False
FAIL_REGEX      = re.compile(r"^\s*(fail(ed)?|error|n/?a|none|null)?\s*$", re.I)

# BERTScore device/batch settings
# Use GPU if available; override with env BERTSCORE_DEVICE (e.g., "cuda:1" or "cpu")
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = os.environ.get("BERTSCORE_DEVICE", DEFAULT_DEVICE)
BERT_BATCH_SIZE = int(os.environ.get("BERTSCORE_BATCH", "8"))

# ===================== Helpers =====================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def is_fail(s) -> bool:
    if s is None: return True
    if isinstance(s, float) and np.isnan(s): return True
    s = str(s)
    return bool(FAIL_REGEX.match(s)) or (s.strip() == "")

def clean_text(s, lower=False) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower() if lower else s

def corpus_bleu(preds, refs):
    try:
        return sacrebleu.corpus_bleu(preds, [refs]).score
    except Exception:
        return float("nan")

def evaluate_one_model(model_name: str) -> pd.DataFrame:
    """Evaluate a single model's report column against GT and return a 1-row summary DataFrame."""
    pred_csv = PRED_PATTERN.format(model=model_name)
    if not os.path.exists(pred_csv):
        print(f"[WARN] Missing prediction CSV for {model_name}: {pred_csv}")
        return pd.DataFrame()

    if not os.path.exists(GT_CSV):
        raise FileNotFoundError(f"Ground truth CSV not found: {GT_CSV}")

    pred = pd.read_csv(pred_csv)
    gt   = pd.read_csv(GT_CSV)

    # Basic column checks
    for df, name, cols in [(pred, "pred", [ID_COL, PRED_TEXT_COL]),
                           (gt,   "gt",   [ID_COL, GT_TEXT_COL])]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} CSV missing columns: {missing}")

    pred2 = pred[[ID_COL, PRED_TEXT_COL]].copy().rename(columns={PRED_TEXT_COL: "pred_text"})
    gt2   = gt[[ID_COL, GT_TEXT_COL]].copy().rename(columns={GT_TEXT_COL: "gt_text"})

    merged = pred2.merge(gt2, on=ID_COL, how="left", validate="m:1")

    # Filter unusable rows
    merged = merged[~merged["pred_text"].apply(is_fail)].copy()
    merged["gt_text"] = merged["gt_text"].fillna("").astype(str)
    merged = merged[merged["gt_text"].str.strip() != ""].copy()

    merged["pred_eval"] = merged["pred_text"].map(lambda x: clean_text(x, LOWERCASE))
    merged["gt_eval"]   = merged["gt_text"].map(lambda x: clean_text(x, LOWERCASE))

    merged = merged[(merged["pred_eval"] != "") & (merged["gt_eval"] != "")]
    if merged.empty:
        print(f"[INFO] {model_name}: Nothing to evaluate after filtering.")
        return pd.DataFrame([{
            "model": model_name,
            "n_pairs": 0,
            "bleu_corpus": np.nan,
            "rouge1_f1_mean": np.nan,
            "rougeL_f1_mean": np.nan,
            "berts_f1_mean": np.nan
        }])

    preds = merged["pred_eval"].tolist()
    refs  = merged["gt_eval"].tolist()
    n_pairs = len(merged)

    # ROUGE F1 (pairwise then mean)
    rscorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r1_f, rL_f = [], []
    for p, r in zip(preds, refs):
        sc = rscorer.score(r, p)   # note: reference first
        r1_f.append(sc["rouge1"].fmeasure)
        rL_f.append(sc["rougeL"].fmeasure)
    rouge1_f1_mean = float(np.mean(r1_f))
    rougeL_f1_mean = float(np.mean(rL_f))

    # SacreBLEU (corpus)
    bleu_corpus = float(corpus_bleu(preds, refs))

    # BERTScore (F1)
    # You can also pass `lang=LANG`, but we specify the model explicitly here.
    P, R, F1 = bert_score(
        preds, refs,
        model_type=BERTSCORE_MODEL,
        rescale_with_baseline=RESCALE_BERT,
        device=DEVICE,
        batch_size=BERT_BATCH_SIZE
    )
    berts_f1_mean = float(F1.mean().item())

    return pd.DataFrame([{
        "model": model_name,
        "n_pairs": n_pairs,
        "bleu_corpus": bleu_corpus,
        "rouge1_f1_mean": rouge1_f1_mean,
        "rougeL_f1_mean": rougeL_f1_mean,
        "berts_f1_mean": berts_f1_mean
    }])

def main():
    ensure_dir(OUT_DIR)

    all_rows = []
    for m in MODELS:
        print(f"==== Evaluating: {m} ====")
        try:
            df_sum = evaluate_one_model(m)
            if not df_sum.empty:
                # Save per-model summary
                out_path = os.path.join(OUT_DIR, f"Task6_{m}_metrics_summary.csv")
                df_sum.drop(columns=["model"], errors="ignore").to_csv(out_path, index=False)
                print(f"[OK] {m} -> {out_path}")
                all_rows.append(df_sum)
            else:
                print(f"[INFO] {m}: Skipped (no data).")
        except Exception as e:
            print(f"[ERROR] {m}: {e}")

    # Save overall summary across models
    if all_rows:
        overall = pd.concat(all_rows, ignore_index=True)
        overall_path = os.path.join(OUT_DIR, "overall_report_metrics.csv")
        overall.to_csv(overall_path, index=False)
        print(f"[OK] Overall -> {overall_path}")
    else:
        print("[INFO] No summaries produced.")

if __name__ == "__main__":
    main()
