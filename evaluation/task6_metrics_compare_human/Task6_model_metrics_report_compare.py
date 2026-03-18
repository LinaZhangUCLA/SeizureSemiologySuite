
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

def calculate_metrics(PRED_CSV,model):
    pred = pd.read_csv(PRED_CSV)
    gt   = pd.read_csv(GT_CSV)

    file_list = [

    "A0002@5-13-2021@UA6693LK@sz_v1_1.mp4",
    "A0004@5-21-2019@DA0012TK@sz_v2_1.mp4",
    "B0001@9-15-2022@DA1322M1@sz_v1_1.mp4",
    "D0002@5-29-2014@TA76406Z@nes_v3_1.mp4",
    "E0003@6-27-2022@KA96022I@nes_v1_1.mp4",
    "F0002@4-26-2018@DA0010PI@sz_v1_1.mp4",
    "H0003@6-2-2022@KA9601YM@sz_v2_1.mp4",
    "I0002@5-9-2019@DA0012RS@nes_v2_1.mp4",
    "I0005@6-8-2022@KA960202@nes_v1_1.mp4",        
    "L0001@5-31-2022@KA9601XM@sz_v1_1.mp4",
    "L0003@10-28-2014@TA7640V1@nes_v1_1.mp4",
    "N0007@8-26-2014@VA7631F0@sz_v2_1.mp4",
    "N0008@2-1-2016@KA333KWO@nes_v1_1.mp4",
    "N0011@4-1-2021@DA00161F@sz_v1_1.mp4",
    "N0011@4-1-2021@DA00161K@sz_v1_1.mp4",
    "N0012@10-10-2019@DA0013ME@sz_v4_1.mp4",
    "R0007@3-7-2016@CA4444H1@nes_v1_1.mp4",
    "R0015@3-17-2022@KA9601DC@sz_v1_1.mp4",
    "S0006@11-25-2014@TA7640ZP@nes_v1_1.mp4",
    "R0011@9-22-2020@DA00158B@nes_v1_1.mp4",
    ]
    pred = pred[pred.iloc[:, 0].isin(file_list)].reset_index(drop=True)

    if len(pred)!=20:
        raise ValueError(f"report file length error {len(pred)}")


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
    
    if len(merged)!=20:
        raise ValueError(f"merged length error {len(merged)}")
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
    # rouge1_f1_mean = float(np.mean(r1_f_list))
    # rougeL_f1_mean = float(np.mean(rL_f_list))

    # SacreBLEU（corpus）
    # bleu_corpus = float(corpus_bleu(preds, refs))


    import sacrebleu

    def pair_bleu_scores(preds, refs):
        scores = []
        for pred, ref in zip(preds, refs):
            result = sacrebleu.sentence_bleu(pred, [ref])
            scores.append(result.score)  # 0~100
        return scores
    
    bleu_corpus = pair_bleu_scores(preds, refs)



    DEVICE = 'cuda:3'  
    torch.cuda.set_device(3)

    from bert_score import score
    P, R, F1 = score(preds, refs, model_type=BERTSCORE_MODEL, rescale_with_baseline=RESCALE_BERT, batch_size=8,device=DEVICE)

    file_name = merged[ID_COL].tolist()
    if len(file_name)!=20 or len(F1)!=20 or len(r1_f_list) !=20 or len(rL_f_list)!=20 or len(bleu_corpus)!=20:
        raise ValueError(f"metrics length error {len(F1)}")

    row_metrics = []
    for f, b, r1,rl,f1  in zip(file_name, bleu_corpus, r1_f_list, rL_f_list, F1):
        # print(type(b))
        # # break

        

        # print(type(r1))
        # print(type(rl))
        # print(type(f1))
        # break


        row_metrics.append({
            "model": model,
            "file_name": f,
            "bleu_corpus": round( float(b),2),
            "rouge1_f1": round(float(r1),2),
            "rougeL_f1": round(float(rl),2),
            "berts_f1": round(float(f1),2),
        })
        #print(row_metrics[0])
    return row_metrics
    # # BERTScore（F1）
    # P, R, F1 = bert_score(preds, refs, lang=LANG,
    #                       model_type=BERTSCORE_MODEL,
    #                       rescale_with_baseline=RESCALE_BERT)
    # berts_f1_mean = float(F1.mean().item())
    # print(n_pairs,PRED_CSV)
    # return({
    #     "bleu_corpus": round(bleu_corpus,2),
    #     "rouge1_f1": round(rouge1_f1_mean,2),
    #     "rougeL_f1": round(rougeL_f1_mean,2),
    #     "berts_f1": round(berts_f1_mean,2)
    # })

if __name__ == "__main__":

    MODELS = [
        # "InternVL3_5-8B",
        # "Qwen2.5-VL-7B-Instruct",
        # 'Qwen3-VL-8B-Instruct',
        "InternVL3_5-38B",
        # "Qwen2.5-VL-32B-Instruct",
        # 'Qwen3-VL-32B-Instruct',
        "Qwen2.5-VL-72B-Instruct",
        
        "Qwen2.5-Omni-7B",
        # "Qwen3-Omni-30B-A3B-Instruct",
        # "Lingshu-32B",
    ]    


       
       

    metric_rows = []
    for model in MODELS:
        input_csv = f"{base_dir}/result/vlm_inference/{model}/Task6_{model}_all_merged.csv"
        metric_rows = metric_rows + calculate_metrics(input_csv,model)

        # model_metrics["model"] = model
        # metric_rows.append(model_metrics)

    out_df = pd.DataFrame(metric_rows, columns=["model",'file_name',"bleu_corpus", "rouge1_f1","rougeL_f1","berts_f1"])
    out_path = f"{base_dir}metrics/compare_with_human_score.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print("save to ", out_path)

        
        
  
