# Download the bertscore model (run in terminal; replace <HF_CACHE_DIR> with your path):
# huggingface-cli download microsoft/deberta-xlarge-mnli --cache-dir <HF_CACHE_DIR>

import os, re
import numpy as np
import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score

os.environ["TRANSFORMERS_CACHE"] = "/mnt/SSD3/xinyi/hf_cache"

PRED_CSV = "/mnt/SSD3/xinyi/benchmark/task2_metrics/Task2_llmmerge_results/Task1_InternVL3_5-8B_all_merged_llmmerge.csv"  
GT_CSV   = "/mnt/SSD3/xinyi/benchmark/task2_metrics/task12_annotation.csv"   
OUT_DIR  = "/mnt/SSD3/xinyi/benchmark/task2_metrics/Task2_llmmerge_results"
ID_COL   = "file_name"

LOWERCASE = True
LANG = "en"                
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"    
RESCALE_BERT = False
FAIL_REGEX = re.compile(r"^\s*(fail(ed)?|error|n/?a|none|null)?\s*$", re.I)


os.makedirs(OUT_DIR, exist_ok=True)
pred = pd.read_csv(PRED_CSV)
gt   = pd.read_csv(GT_CSV)


J_PREFIX = "justification_for_"
feat_cols = [c for c in pred.columns if c.startswith(J_PREFIX)]
if not feat_cols:
    raise ValueError(f"No columns start with '{J_PREFIX}' in prediction CSV.")

def feat_name(col): return col[len(J_PREFIX):]

def to_long(df, id_col, feat_cols, label):
    parts = []
    for c in feat_cols:
        tmp = df[[id_col, c]].copy()
        tmp.columns = [id_col, label]
        tmp["feature"] = feat_name(c)
        parts.append(tmp)
    return pd.concat(parts, ignore_index=True)

pred_long = to_long(pred, ID_COL, feat_cols, "pred_text")
gt_long   = to_long(gt,   ID_COL, feat_cols, "gt_text")

merged = pred_long.merge(gt_long, on=[ID_COL, "feature"], how="left", validate="m:1")

def is_fail(s):
    if s is None: return True
    if isinstance(s, float) and np.isnan(s): return True
    s = str(s)
    return bool(FAIL_REGEX.match(s)) or (s.strip() == "")

merged = merged[~merged["pred_text"].apply(is_fail)].copy()
merged["gt_text"] = merged["gt_text"].fillna("").astype(str)
merged = merged[merged["gt_text"].str.strip() != ""].copy()

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

def corpus_bleu(preds, refs):
    try: return sacrebleu.corpus_bleu(preds, [refs]).score
    except Exception: return float("nan")

rscorer = rouge_scorer.RougeScorer(["rouge1","rougeL"], use_stemmer=True)

rows = []
for f, grp in merged.groupby("feature"):
    preds = grp["pred_eval"].tolist()
    refs  = grp["gt_eval"].tolist()
    n = len(grp)

    r1, rL = [], []
    for c, r in zip(preds, refs):
        sc = rscorer.score(r, c)
        r1.append(sc["rouge1"].fmeasure)
        rL.append(sc["rougeL"].fmeasure)

    _, _, F1 = bert_score(preds, refs, lang=LANG,
                          model_type=BERTSCORE_MODEL,
                          rescale_with_baseline=RESCALE_BERT)
    rows.append({
        "feature": f,
        "n_pairs": n,
        "bleu_corpus": corpus_bleu(preds, refs),
        "rouge1_f1_mean": float(np.mean(r1)),
        "rougeL_f1_mean": float(np.mean(rL)),
        "berts_f1_mean": float(F1.numpy().mean()),
    })

feat_sum = pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)

numeric_cols = ["n_pairs", "bleu_corpus", "rouge1_f1_mean", "rougeL_f1_mean", "berts_f1_mean"]

avg_row = {"feature": "AVG"}
avg_row["n_pairs"] = int(round(feat_sum["n_pairs"].mean()))

for col in numeric_cols[1:]:
    avg_row[col] = float(feat_sum[col].mean())

feat_sum = pd.concat([feat_sum, pd.DataFrame([avg_row])], ignore_index=True)

feat_sum.to_csv(os.path.join(OUT_DIR, "Task1_InternVL3_5-8B_llmmerge_feature_summary.csv"), index=False)
print(f"Saved -> {os.path.join(OUT_DIR, 'Task1_InternVL3_5-8B_llmmerge_feature_summary.csv')}")

# python /mnt/SSD3/xinyi/benchmark/task2_metrics/metrics.py
