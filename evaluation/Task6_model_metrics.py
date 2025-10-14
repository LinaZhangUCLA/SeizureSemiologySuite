import os, re
import numpy as np
import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import torch

# To avoid downloading the model from the internet at runtime, please pre-cache the model on the server into the specified directory:
# mkdir -p /mnt/SSD3/xinyi/hf_cache
# huggingface-cli download microsoft/deberta-xlarge-mnli \
#     --local-dir /mnt/SSD3/xinyi/hf_cache/deberta_mnli

os.environ["TRANSFORMERS_CACHE"] = "/mnt/SSD3/xinyi/hf_cache"

# ===== Figuration =====
PRED_CSV = "/mnt/SSD3/xinyi/benchmark/task6_metrics/Task6_llmmerge/Task6_Qwen2.5-VL-72B-Instruct_all_merged_llmmerge.csv"  
GT_CSV   = "/mnt/SSD3/xinyi/benchmark/task6_metrics/task6_annotation_new.csv"                                         
OUT_DIR  = "/mnt/SSD3/xinyi/benchmark/task6_metrics/Task6_llmmerge_results"
ID_COL   = "file_name"
PRED_TEXT_COL = "report"
GT_TEXT_COL   = "report"

LOWERCASE = True                      
LANG = "en"                          
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli" 
RESCALE_BERT = False                  

FAIL_REGEX = re.compile(r"^\s*(fail(ed)?|error|n/?a|none|null)?\s*$", re.I)
# ==================================

os.makedirs(OUT_DIR, exist_ok=True)

def is_fail(s):
    if s is None: return True
    if isinstance(s, float) and np.isnan(s): return True
    s = str(s)
    return bool(FAIL_REGEX.match(s)) or (s.strip() == "")

def clean_text(s, lower=False):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    if lower:
        s = s.lower()
    return s

def corpus_bleu(preds, refs):
    try:
        return sacrebleu.corpus_bleu(preds, [refs]).score
    except Exception:
        return float("nan")


pred = pd.read_csv(PRED_CSV)
gt   = pd.read_csv(GT_CSV)


for df, name, cols in [(pred, "pred", [ID_COL, PRED_TEXT_COL]),
                       (gt,   "gt",   [ID_COL, GT_TEXT_COL])]:
    lack = [c for c in cols if c not in df.columns]
    if lack:
        raise ValueError(f"{name} CSV 缺少列: {lack}")

pred2 = pred[[ID_COL, PRED_TEXT_COL]].copy().rename(columns={PRED_TEXT_COL: "pred_text"})
gt2   = gt[[ID_COL, GT_TEXT_COL]].copy().rename(columns={GT_TEXT_COL: "gt_text"})

merged = pred2.merge(gt2, on=ID_COL, how="left", validate="m:1")

merged = merged[~merged["pred_text"].apply(is_fail)].copy()
merged["gt_text"] = merged["gt_text"].fillna("").astype(str)
merged = merged[merged["gt_text"].str.strip() != ""].copy()

merged["pred_eval"] = merged["pred_text"].map(lambda x: clean_text(x, LOWERCASE))
merged["gt_eval"]   = merged["gt_text"].map(lambda x: clean_text(x, LOWERCASE))

merged = merged[(merged["pred_eval"] != "") & (merged["gt_eval"] != "")]
if merged.empty:
    print("Nothing to evaluate after filtering.")

    pd.DataFrame([{
        "n_pairs": 0,
        "bleu_corpus": np.nan,
        "rouge1_f1_mean": np.nan,
        "rougeL_f1_mean": np.nan,
        "berts_f1_mean": np.nan
    }]).to_csv(os.path.join(OUT_DIR, "report_metrics_summary.csv"), index=False)
    raise SystemExit

preds = merged["pred_eval"].tolist()
refs  = merged["gt_eval"].tolist()
n_pairs = len(merged)

# ROUGE（F1）
rscorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
r1_f_list, rL_f_list = [], []
for p, r in zip(preds, refs):
    sc = rscorer.score(r, p)   
    r1_f_list.append(sc["rouge1"].fmeasure)
    rL_f_list.append(sc["rougeL"].fmeasure)
rouge1_f1_mean = float(np.mean(r1_f_list))
rougeL_f1_mean = float(np.mean(rL_f_list))

# SacreBLEU（corpus）
bleu_corpus = float(corpus_bleu(preds, refs))


DEVICE = 'cuda:0'  
torch.cuda.set_device(0)

from bert_score import score
P, R, F1 = score(preds, refs, model_type=BERTSCORE_MODEL, rescale_with_baseline=RESCALE_BERT, batch_size=8,device=DEVICE)

# # BERTScore（F1）
# P, R, F1 = bert_score(preds, refs, lang=LANG,
#                       model_type=BERTSCORE_MODEL,
#                       rescale_with_baseline=RESCALE_BERT)
berts_f1_mean = float(F1.mean().item())


summary_path = os.path.join(OUT_DIR, "Task6_Qwen2.5-VL-72B-Instruct_metrics_summary.csv")
pd.DataFrame([{
    "n_pairs": n_pairs,
    "bleu_corpus": bleu_corpus,
    "rouge1_f1_mean": rouge1_f1_mean,
    "rougeL_f1_mean": rougeL_f1_mean,
    "berts_f1_mean": berts_f1_mean
}]).to_csv(summary_path, index=False)
print(f"Saved summary -> {summary_path}")
