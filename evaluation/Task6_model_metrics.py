
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

#os.environ["TRANSFORMERS_CACHE"] = "/mnt/SSD3/xinyi/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/SSD3/lina/bertmodel"


# ===== Figuration =====

base_dir = '/home/lina/ssb/SeizureSemiologyBench/'
GT_CSV   = f"{base_dir}result/ground_truth/task6_report_annotation.csv"                                         

ID_COL   = "file_name"
PRED_TEXT_COL = "report"
GT_TEXT_COL   = "report"

LOWERCASE = True                      
LANG = "en"                          
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli" 
RESCALE_BERT = False                  

FAIL_REGEX = re.compile(r"^\s*(fail(ed)?|error|n/?a|none|null)?\s*$", re.I)
# ==================================



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

def calculate_metrics(PRED_CSV):
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


    DEVICE = 'cuda:3'  
    torch.cuda.set_device(3)

    from bert_score import score
    P, R, F1 = score(preds, refs, model_type=BERTSCORE_MODEL, rescale_with_baseline=RESCALE_BERT, batch_size=8,device=DEVICE)

    # # BERTScore（F1）
    # P, R, F1 = bert_score(preds, refs, lang=LANG,
    #                       model_type=BERTSCORE_MODEL,
    #                       rescale_with_baseline=RESCALE_BERT)
    berts_f1_mean = float(F1.mean().item())
    print(n_pairs,PRED_CSV)
    return({
        "bleu_corpus": round(bleu_corpus,2),
        "rouge1_f1": round(rouge1_f1_mean,2),
        "rougeL_f1": round(rougeL_f1_mean,2),
        "berts_f1": round(berts_f1_mean,2)
    })

if __name__ == "__main__":

    MODELS = [
        "InternVL3_5-8B",
        # #"InternVL3_5-38B",
        # "Qwen2.5-VL-7B-Instruct",
        # "Qwen2.5-VL-32B-Instruct",
        # "Qwen2.5-VL-72B-Instruct",
        # #"Lingshu-32B",
        # "Qwen2.5-Omni-7B",
        # 'Qwen3-VL-8B-Instruct',
        # 'Qwen3-VL-32B-Instruct',
        # "Qwen3-Omni-30B-A3B-Instruct"
    ]    
    metric_rows = []
    for model in MODELS:
        input_csv = f"{base_dir}/result/vlm_inference/{model}/Task6_{model}_all_merged.csv"
        model_metrics = calculate_metrics(input_csv)

        model_metrics["model"] = model
        metric_rows.append(model_metrics)

    out_df = pd.DataFrame(metric_rows, columns=["model","bleu_corpus", "rouge1_f1","rougeL_f1","berts_f1"])
    out_path = f"{base_dir}metrics/task6_report_nlp_metrics.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8")

        
        
  
