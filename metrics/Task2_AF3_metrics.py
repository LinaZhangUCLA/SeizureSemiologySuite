import os, re
import numpy as np
import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# ================== Environment ==================
# Prefer environment variable; fallback to a placeholder (replace <HF_CACHE_DIR>).
os.environ["TRANSFORMERS_CACHE"] = os.environ.get("TRANSFORMERS_CACHE", "<HF_CACHE_DIR>")

# ================== Config (paths hidden) ==================
# Use environment variable PROJECT_BASE_DIR or replace <BASE_DIR> with your own base directory.
BASE_DIR = os.environ.get("PROJECT_BASE_DIR", "<BASE_DIR>")

# Input / output files assembled from BASE_DIR; replace the filenames if needed.
PRED_CSV = os.path.join(BASE_DIR, "task2_metrics", "Task1_AF3_Full_Results.csv")      # prediction CSV
GT_CSV   = os.path.join(BASE_DIR, "task2_metrics", "task12_annotation.csv")           # ground truth CSV
OUT_DIR  = os.path.join(BASE_DIR, "task2_metrics", "Task2_llmmerge_results")

# Column holding the sample ID (file name)
ID_COL = "file_name"

# ================== Evaluation Settings ==================
LOWERCASE = True
LANG = "en"  # BERTScore language
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"
RESCALE_BERT = False
FAIL_REGEX = re.compile(r"^\s*(fail(ed)?|error|n/?a|none|null)?\s*$", re.I)

# Force-align file name suffix for both prediction and GT sides
ID_FORCE_EXT = ".wav"
# =========================================================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def normalize_to_ext(s: str, ext: str = ".wav") -> str:
    """
    Normalize a file name to ensure it ends with `ext`:
      - If it already ends with `ext` (case-insensitive), rewrite it to lower-case `ext`.
      - If it ends with another extension, replace that suffix with `ext`.
      - If it has no extension, append `ext`.
    Only the trailing extension is modified; dots in the middle are preserved.
    """
    s = ("" if s is None else str(s)).strip()
    if s == "":
        return s
    # Already ends with target ext (case-insensitive) -> standardize to lowercase ext
    if re.search(rf"(?i){re.escape(ext)}$", s):
        return re.sub(rf"(?i){re.escape(ext)}$", ext, s)
    # Has some other extension -> replace the trailing .xxx with target ext
    if re.search(r"\.[A-Za-z0-9]+$", s):
        return re.sub(r"\.[A-Za-z0-9]+$", ext, s)
    # No extension -> append target ext
    return s + ext

ensure_dir(OUT_DIR)

pred = pd.read_csv(PRED_CSV)
gt   = pd.read_csv(GT_CSV)

# -------- Normalize the ID column on both sides to the same extension (.wav) --------
if ID_COL not in pred.columns:
    raise ValueError(f"Prediction CSV missing id column: {ID_COL}")
if ID_COL not in gt.columns:
    raise ValueError(f"GT CSV missing id column: {ID_COL}")

pred = pred.copy()
gt   = gt.copy()
pred[ID_COL] = pred[ID_COL].astype(str).map(lambda x: normalize_to_ext(x, ID_FORCE_EXT))
gt[ID_COL]   = gt[ID_COL].astype(str).map(lambda x: normalize_to_ext(x, ID_FORCE_EXT))
# -----------------------------------------------------------------------------------

# Discover justification columns
J_PREFIX = "justification_for_"
feat_cols = [c for c in pred.columns if c.startswith(J_PREFIX)]
if not feat_cols:
    raise ValueError(f"No columns start with '{J_PREFIX}' in prediction CSV.")

def feat_name(col: str) -> str:
    return col[len(J_PREFIX):]

# Wide -> long helper (only needed columns)
def to_long(df: pd.DataFrame, id_col: str, feat_cols: list[str], label: str) -> pd.DataFrame:
    parts = []
    for c in feat_cols:
        tmp = df[[id_col, c]].copy()
        tmp.columns = [id_col, label]
        tmp["feature"] = feat_name(c)
        parts.append(tmp)
    return pd.concat(parts, ignore_index=True)

pred_long = to_long(pred, ID_COL, feat_cols, "pred_text")
gt_long   = to_long(gt,   ID_COL, feat_cols, "gt_text")

# Left-join with predictions as the primary table
merged = pred_long.merge(gt_long, on=[ID_COL, "feature"], how="left", validate="m:1")

# Filter out failed/blank predictions, and rows with blank GT
def is_fail(s) -> bool:
    if s is None: return True
    if isinstance(s, float) and np.isnan(s): return True
    s = str(s)
    return bool(FAIL_REGEX.match(s)) or (s.strip() == "")

merged = merged[~merged["pred_text"].apply(is_fail)].copy()
merged["gt_text"] = merged["gt_text"].fillna("").astype(str)
merged = merged[merged["gt_text"].str.strip() != ""].copy()

# Normalize case if required
if LOWERCASE:
    merged["pred_eval"] = merged["pred_text"].astype(str).str.strip().str.lower()
    merged["gt_eval"]   = merged["gt_text"].astype(str).str.strip().str.lower()
else:
    merged["pred_eval"] = merged["pred_text"].astype(str).str.strip()
    merged["gt_eval"]   = merged["gt_text"].astype(str).str.strip()

if merged.empty:
    print("Nothing to evaluate after filtering.")
    pd.DataFrame(columns=[
        "feature","n_pairs","bleu_corpus","rouge1_f1_mean","rougeL_f1_mean","berts_f1_mean"
    ]).to_csv(os.path.join(OUT_DIR, "feature_summary.csv"), index=False)
    raise SystemExit

# Metrics helpers
def corpus_bleu(preds, refs):
    try:
        return sacrebleu.corpus_bleu(preds, [refs]).score
    except Exception:
        return float("nan")

rscorer = rouge_scorer.RougeScorer(["rouge1","rougeL"], use_stemmer=True)

# Aggregate per feature (ROUGE/BERT are computed per pair, then averaged)
rows = []
for f, grp in merged.groupby("feature"):
    preds = grp["pred_eval"].tolist()
    refs  = grp["gt_eval"].tolist()
    n = len(grp)

    # ROUGE F1 mean
    r1, rL = [], []
    for c, r in zip(preds, refs):
        sc = rscorer.score(r, c)
        r1.append(sc["rouge1"].fmeasure)
        rL.append(sc["rougeL"].fmeasure)

    # BERTScore F1 mean
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

# Append an average row at the end (macro mean across features)
numeric_cols = ["n_pairs", "bleu_corpus", "rouge1_f1_mean", "rougeL_f1_mean", "berts_f1_mean"]
avg_row = {"feature": "AVG_18"}
# `n_pairs` is usually similar across rows; we take the mean and round. Use sum() if you prefer total pairs.
avg_row["n_pairs"] = int(round(feat_sum["n_pairs"].mean()))
for col in numeric_cols[1:]:
    avg_row[col] = float(feat_sum[col].mean())

feat_sum = pd.concat([feat_sum, pd.DataFrame([avg_row])], ignore_index=True)

out_path = os.path.join(OUT_DIR, "Task1_AF3_llmmerge_feature_summary.csv")
feat_sum.to_csv(out_path, index=False)
print(f"Saved -> {out_path}")
