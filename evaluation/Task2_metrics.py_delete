# Download the bertscore model (run in terminal; replace <HF_CACHE_DIR> with your path):
# huggingface-cli download microsoft/deberta-xlarge-mnli --cache-dir <HF_CACHE_DIR>

import os, re, glob
import numpy as np
import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer
import torch
from transformers import AutoTokenizer, AutoModel

# Configuration
VLM_INFERENCE_DIR = "result/vlm_inference"
GT_CSV = "result/ground_truth/task12_annotation.csv"
OUT_DIR = "metrics/Task2_feature_metrics"
ID_COL = "file_name"
LOWERCASE = True
LANG = "en"

# Local model path
LOCAL_MODEL_PATH = '/Users/eehan/Desktop/deberta-xlarge-mnli'

# Number of layers to use (DeBERTa-xlarge has 24 layers, typically use layer 9 for similarity)
NUM_LAYERS = 9

FAIL_REGEX = re.compile(r"^\s*(fail(ed)?|error|n/?a|none|null)?\s*$", re.I)

# Create output directory
os.makedirs(OUT_DIR, exist_ok=True)

# Find all CSV files matching the pattern
csv_pattern = os.path.join(VLM_INFERENCE_DIR, "**/*all_merged_llmmerge.csv")
pred_csv_files = glob.glob(csv_pattern, recursive=True)

if not pred_csv_files:
    print(f"No CSV files found matching pattern: {csv_pattern}")
    exit(1)

print(f"Found {len(pred_csv_files)} CSV file(s) to process:")
for f in pred_csv_files:
    print(f"  - {f}")
print()

# Load model once (will be reused for all files)
print(f"Loading model from: {LOCAL_MODEL_PATH}")
if not os.path.exists(LOCAL_MODEL_PATH):
    raise FileNotFoundError(f"Model directory not found: {LOCAL_MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModel.from_pretrained(LOCAL_MODEL_PATH)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()
print(f"Model loaded successfully on {device}!\n")

# Load ground truth once
gt = pd.read_csv(GT_CSV)


def feat_name(col):
    J_PREFIX = "justification_for_"
    return col[len(J_PREFIX):]


def to_long(df, id_col, feat_cols, label):
    parts = []
    for c in feat_cols:
        tmp = df[[id_col, c]].copy()
        tmp.columns = [id_col, label]
        tmp["feature"] = feat_name(c)
        parts.append(tmp)
    return pd.concat(parts, ignore_index=True)


def is_fail(s):
    if s is None:
        return True
    if isinstance(s, float) and np.isnan(s):
        return True
    s = str(s)
    return bool(FAIL_REGEX.match(s)) or (s.strip() == "")


def corpus_bleu(preds, refs):
    try:
        return sacrebleu.corpus_bleu(preds, [refs]).score
    except Exception:
        return float("nan")


def compute_bertscore(cands, refs, model, tokenizer, device, num_layers=9, batch_size=64):
    """
    Compute BERTScore F1 using local model
    """
    all_f1 = []
    for i in range(0, len(cands), batch_size):
        batch_cands = cands[i:i + batch_size]
        batch_refs = refs[i:i + batch_size]

        cand_inputs = tokenizer(batch_cands, padding=True, truncation=True,
                                return_tensors='pt', max_length=512).to(device)
        ref_inputs = tokenizer(batch_refs, padding=True, truncation=True,
                               return_tensors='pt', max_length=512).to(device)

        with torch.no_grad():
            cand_outputs = model(**cand_inputs, output_hidden_states=True)
            ref_outputs = model(**ref_inputs, output_hidden_states=True)

            cand_embs = cand_outputs.hidden_states[num_layers]
            ref_embs = ref_outputs.hidden_states[num_layers]

            for j in range(len(batch_cands)):
                cand_mask = cand_inputs['attention_mask'][j].bool()
                ref_mask = ref_inputs['attention_mask'][j].bool()

                cand_emb = cand_embs[j][cand_mask]
                ref_emb = ref_embs[j][ref_mask]

                sim_matrix = torch.nn.functional.cosine_similarity(
                    cand_emb.unsqueeze(1), ref_emb.unsqueeze(0), dim=2
                )

                precision = sim_matrix.max(dim=1)[0].mean()
                recall = sim_matrix.max(dim=0)[0].mean()

                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = torch.tensor(0.0)

                all_f1.append(f1.item())

    return all_f1


def process_csv(pred_csv_path, gt, model, tokenizer, device):
    """Process a single prediction CSV file"""
    pred_filename = os.path.basename(pred_csv_path)
    match = re.search(r'(Qwen[\w\.-]+|InternVL[\w\.-]+)', pred_filename, re.IGNORECASE)
    if match:
        model_identifier = match.group(1)
    else:
        folder_name = os.path.basename(os.path.dirname(pred_csv_path))
        model_identifier = folder_name

    model_identifier = re.sub(r'-Instruct', '', model_identifier, flags=re.IGNORECASE)

    # 生成输出文件名
    output_filename = f"{model_identifier}_feature_summary.csv"
    out_file = os.path.join(OUT_DIR, output_filename)

    print(f"Processing: {pred_csv_path}")
    print(f"  Model identifier: {model_identifier}")
    print(f"  Output file: {out_file}")

    # 检查输出文件是否已存在
    if os.path.exists(out_file):
        print(f"  ✓ Output file already exists, skipping.\n")
        return

    existing_files = glob.glob(os.path.join(OUT_DIR, "*_feature_summary.csv"))
    base_name = model_identifier.lower().replace('-', '_').replace('.', '')
    for existing in existing_files:
        existing_base = os.path.basename(existing).replace('_feature_summary.csv', '').lower().replace('-', '_').replace('.', '')
        if base_name == existing_base:
            print(f"  ✓ Similar output file already exists ({os.path.basename(existing)}), skipping.\n")
            return

    try:
        pred = pd.read_csv(pred_csv_path)
    except Exception as e:
        print(f"  ✗ Error reading CSV: {e}\n")
        return

    J_PREFIX = "justification_for_"
    feat_cols = [c for c in pred.columns if c.startswith(J_PREFIX)]

    if not feat_cols:
        print(f"  ✗ No columns start with '{J_PREFIX}', skipping.\n")
        return

    pred_long = to_long(pred, ID_COL, feat_cols, "pred_text")
    gt_long = to_long(gt, ID_COL, feat_cols, "gt_text")

    merged = pred_long.merge(gt_long, on=[ID_COL, "feature"], how="left", validate="m:1")

    merged = merged[~merged["pred_text"].apply(is_fail)].copy()
    merged["gt_text"] = merged["gt_text"].fillna("").astype(str)
    merged = merged[merged["gt_text"].str.strip() != ""].copy()

    if LOWERCASE:
        merged["pred_eval"] = merged["pred_text"].astype(str).str.strip().str.lower()
        merged["gt_eval"] = merged["gt_text"].astype(str).str.strip().str.lower()
    else:
        merged["pred_eval"] = merged["pred_text"].astype(str).str.strip()
        merged["gt_eval"] = merged["gt_text"].astype(str).str.strip()

    if merged.empty:
        print("  ✗ Nothing to evaluate after filtering.\n")
        pd.DataFrame(columns=[
            "feature", "n_pairs", "bleu_corpus", "rouge1_f1_mean", "rougeL_f1_mean", "berts_f1_mean"
        ]).to_csv(out_file, index=False)
        return

    rscorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    rows = []
    for f, grp in merged.groupby("feature"):
        preds = grp["pred_eval"].tolist()
        refs = grp["gt_eval"].tolist()
        n = len(grp)

        r1, rL = [], []
        for c, r in zip(preds, refs):
            sc = rscorer.score(r, c)
            r1.append(sc["rouge1"].fmeasure)
            rL.append(sc["rougeL"].fmeasure)

        print(f"  Computing metrics for feature: {f} ({n} pairs)...")

        f1_scores = compute_bertscore(preds, refs, model, tokenizer, device, NUM_LAYERS)

        rows.append({
            "feature": f,
            "n_pairs": n,
            "bleu_corpus": corpus_bleu(preds, refs),
            "rouge1_f1_mean": float(np.mean(r1)),
            "rougeL_f1_mean": float(np.mean(rL)),
            "berts_f1_mean": float(np.mean(f1_scores)),
        })

    feat_sum = pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)

    numeric_cols = ["n_pairs", "bleu_corpus", "rouge1_f1_mean", "rougeL_f1_mean", "berts_f1_mean"]
    avg_row = {"feature": "AVG"}
    avg_row["n_pairs"] = int(round(feat_sum["n_pairs"].mean()))
    for col in numeric_cols[1:]:
        avg_row[col] = float(feat_sum[col].mean())

    feat_sum = pd.concat([feat_sum, pd.DataFrame([avg_row])], ignore_index=True)

    feat_sum.to_csv(out_file, index=False)
    print(f"  ✓ Saved -> {out_file}\n")


# Process all CSV files
print("=" * 60)
print("Starting batch processing...")
print("=" * 60 + "\n")

for pred_csv in pred_csv_files:
    try:
        process_csv(pred_csv, gt, model, tokenizer, device)
    except Exception as e:
        print(f"✗ Error processing {pred_csv}: {e}\n")
        continue

print("=" * 60)
print("Batch processing complete!")
print("=" * 60)



#Test code for calculate score
# # Download the bertscore model (run in terminal; replace <HF_CACHE_DIR> with your path):
# # huggingface-cli download microsoft/deberta-xlarge-mnli --cache-dir <HF_CACHE_DIR>

# import os, re
# import numpy as np
# import pandas as pd
# import sacrebleu
# from rouge_score import rouge_scorer
# import torch
# from transformers import AutoTokenizer, AutoModel
# from bert_score.utils import get_idf_dict, bert_cos_score_idf

# # Configuration
# PRED_CSV = "result/vlm_inference/InternVL3_5-8B/Task12_InternVL3_5-8B_all_merged_llmmerge.csv"
# GT_CSV = "result/ground_truth/task12_annotation.csv"
# OUT_DIR = "metrics/Task2_llmmerge_results"
# ID_COL = "file_name"
# LOWERCASE = True
# LANG = "en"

# # Local model path
# LOCAL_MODEL_PATH = '/Users/eehan/Desktop/deberta-xlarge-mnli'

# # Verify model path exists
# if not os.path.exists(LOCAL_MODEL_PATH):
#     raise FileNotFoundError(f"Model directory not found: {LOCAL_MODEL_PATH}")

# print(f"Loading model from: {LOCAL_MODEL_PATH}")

# # Load model and tokenizer directly from local path
# tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
# model = AutoModel.from_pretrained(LOCAL_MODEL_PATH)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = model.to(device)
# model.eval()

# print(f"Model loaded successfully on {device}!")

# # Number of layers to use (DeBERTa-xlarge has 24 layers, typically use layer 9 for similarity)
# NUM_LAYERS = 9

# FAIL_REGEX = re.compile(r"^\s*(fail(ed)?|error|n/?a|none|null)?\s*$", re.I)

# os.makedirs(OUT_DIR, exist_ok=True)

# pred = pd.read_csv(PRED_CSV)
# gt = pd.read_csv(GT_CSV)

# J_PREFIX = "justification_for_"
# feat_cols = [c for c in pred.columns if c.startswith(J_PREFIX)]

# if not feat_cols:
#     raise ValueError(f"No columns start with '{J_PREFIX}' in prediction CSV.")

# def feat_name(col):
#     return col[len(J_PREFIX):]

# def to_long(df, id_col, feat_cols, label):
#     parts = []
#     for c in feat_cols:
#         tmp = df[[id_col, c]].copy()
#         tmp.columns = [id_col, label]
#         tmp["feature"] = feat_name(c)
#         parts.append(tmp)
#     return pd.concat(parts, ignore_index=True)

# pred_long = to_long(pred, ID_COL, feat_cols, "pred_text")
# gt_long = to_long(gt, ID_COL, feat_cols, "gt_text")

# merged = pred_long.merge(gt_long, on=[ID_COL, "feature"], how="left", validate="m:1")

# def is_fail(s):
#     if s is None:
#         return True
#     if isinstance(s, float) and np.isnan(s):
#         return True
#     s = str(s)
#     return bool(FAIL_REGEX.match(s)) or (s.strip() == "")

# merged = merged[~merged["pred_text"].apply(is_fail)].copy()
# merged["gt_text"] = merged["gt_text"].fillna("").astype(str)
# merged = merged[merged["gt_text"].str.strip() != ""].copy()

# if LOWERCASE:
#     merged["pred_eval"] = merged["pred_text"].astype(str).str.strip().str.lower()
#     merged["gt_eval"] = merged["gt_text"].astype(str).str.strip().str.lower()
# else:
#     merged["pred_eval"] = merged["pred_text"].astype(str).str.strip()
#     merged["gt_eval"] = merged["gt_text"].astype(str).str.strip()

# if merged.empty:
#     print("Nothing to evaluate after filtering.")
#     pd.DataFrame(columns=[
#         "feature","n_pairs","bleu_corpus","rouge1_f1_mean","rougeL_f1_mean","berts_f1_mean"
#     ]).to_csv(os.path.join(OUT_DIR, "feature_summary.csv"), index=False)
#     raise SystemExit

# def corpus_bleu(preds, refs):
#     try:
#         return sacrebleu.corpus_bleu(preds, [refs]).score
#     except Exception:
#         return float("nan")

# def compute_bertscore(cands, refs, model, tokenizer, device, num_layers=9, batch_size=64):
#     """
#     Compute BERTScore F1 using local model
#     """
#     all_f1 = []
    
#     # Process in batches
#     for i in range(0, len(cands), batch_size):
#         batch_cands = cands[i:i+batch_size]
#         batch_refs = refs[i:i+batch_size]
        
#         # Tokenize
#         cand_inputs = tokenizer(batch_cands, padding=True, truncation=True, 
#                                 return_tensors='pt', max_length=512).to(device)
#         ref_inputs = tokenizer(batch_refs, padding=True, truncation=True, 
#                                return_tensors='pt', max_length=512).to(device)
        
#         with torch.no_grad():
#             # Get embeddings
#             cand_outputs = model(**cand_inputs, output_hidden_states=True)
#             ref_outputs = model(**ref_inputs, output_hidden_states=True)
            
#             # Use specified layer
#             cand_embs = cand_outputs.hidden_states[num_layers]
#             ref_embs = ref_outputs.hidden_states[num_layers]
            
#             # Compute cosine similarity for each pair
#             for j in range(len(batch_cands)):
#                 # Get valid tokens (non-padding)
#                 cand_mask = cand_inputs['attention_mask'][j].bool()
#                 ref_mask = ref_inputs['attention_mask'][j].bool()
                
#                 cand_emb = cand_embs[j][cand_mask]
#                 ref_emb = ref_embs[j][ref_mask]
                
#                 # Compute pairwise cosine similarity
#                 sim_matrix = torch.nn.functional.cosine_similarity(
#                     cand_emb.unsqueeze(1), ref_emb.unsqueeze(0), dim=2
#                 )
                
#                 # Greedy matching
#                 precision = sim_matrix.max(dim=1)[0].mean()
#                 recall = sim_matrix.max(dim=0)[0].mean()
                
#                 if precision + recall > 0:
#                     f1 = 2 * precision * recall / (precision + recall)
#                 else:
#                     f1 = torch.tensor(0.0)
                
#                 all_f1.append(f1.item())
    
#     return all_f1

# rscorer = rouge_scorer.RougeScorer(["rouge1","rougeL"], use_stemmer=True)

# rows = []
# for f, grp in merged.groupby("feature"):
#     preds = grp["pred_eval"].tolist()
#     refs = grp["gt_eval"].tolist()
#     n = len(grp)
    
#     r1, rL = [], []
#     for c, r in zip(preds, refs):
#         sc = rscorer.score(r, c)
#         r1.append(sc["rouge1"].fmeasure)
#         rL.append(sc["rougeL"].fmeasure)
    
#     print(f"Computing BERTScore for feature: {f} with {n} pairs...")
    
#     # Compute BERTScore using local model
#     f1_scores = compute_bertscore(preds, refs, model, tokenizer, device, NUM_LAYERS)
    
#     rows.append({
#         "feature": f,
#         "n_pairs": n,
#         "bleu_corpus": corpus_bleu(preds, refs),
#         "rouge1_f1_mean": float(np.mean(r1)),
#         "rougeL_f1_mean": float(np.mean(rL)),
#         "berts_f1_mean": float(np.mean(f1_scores)),
#     })

# feat_sum = pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)

# numeric_cols = ["n_pairs", "bleu_corpus", "rouge1_f1_mean", "rougeL_f1_mean", "berts_f1_mean"]
# avg_row = {"feature": "AVG"}
# avg_row["n_pairs"] = int(round(feat_sum["n_pairs"].mean()))
# for col in numeric_cols[1:]:
#     avg_row[col] = float(feat_sum[col].mean())

# feat_sum = pd.concat([feat_sum, pd.DataFrame([avg_row])], ignore_index=True)

# out_file = os.path.join(OUT_DIR, "Task1_InternVL3_5-8B_llmmerge_feature_summary.csv")
# feat_sum.to_csv(out_file, index=False)
# print(f"Saved -> {out_file}")




#Single Model csv
# # Download the bertscore model (run in terminal; replace <HF_CACHE_DIR> with your path):
# # huggingface-cli download microsoft/deberta-xlarge-mnli --cache-dir <HF_CACHE_DIR>

# import os, re
# import numpy as np
# import pandas as pd
# import sacrebleu
# from rouge_score import rouge_scorer
# import torch
# from transformers import AutoTokenizer, AutoModel
# from bert_score.utils import get_idf_dict, bert_cos_score_idf

# # Configuration
# PRED_CSV = "result/vlm_inference/InternVL3_5-8B/Task12_InternVL3_5-8B_all_merged_llmmerge.csv"
# GT_CSV = "result/ground_truth/task12_annotation.csv"
# OUT_DIR = "metrics/Task2_llmmerge_results"
# ID_COL = "file_name"
# LOWERCASE = True
# LANG = "en"

# # Local model path
# LOCAL_MODEL_PATH = '/Users/eehan/Desktop/deberta-xlarge-mnli'

# # Verify model path exists
# if not os.path.exists(LOCAL_MODEL_PATH):
#     raise FileNotFoundError(f"Model directory not found: {LOCAL_MODEL_PATH}")

# print(f"Loading model from: {LOCAL_MODEL_PATH}")

# # Load model and tokenizer directly from local path
# tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
# model = AutoModel.from_pretrained(LOCAL_MODEL_PATH)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = model.to(device)
# model.eval()

# print(f"Model loaded successfully on {device}!")

# # Number of layers to use (DeBERTa-xlarge has 24 layers, typically use layer 9 for similarity)
# NUM_LAYERS = 9

# FAIL_REGEX = re.compile(r"^\s*(fail(ed)?|error|n/?a|none|null)?\s*$", re.I)

# os.makedirs(OUT_DIR, exist_ok=True)

# pred = pd.read_csv(PRED_CSV)
# gt = pd.read_csv(GT_CSV)

# J_PREFIX = "justification_for_"
# feat_cols = [c for c in pred.columns if c.startswith(J_PREFIX)]

# if not feat_cols:
#     raise ValueError(f"No columns start with '{J_PREFIX}' in prediction CSV.")

# def feat_name(col):
#     return col[len(J_PREFIX):]

# def to_long(df, id_col, feat_cols, label):
#     parts = []
#     for c in feat_cols:
#         tmp = df[[id_col, c]].copy()
#         tmp.columns = [id_col, label]
#         tmp["feature"] = feat_name(c)
#         parts.append(tmp)
#     return pd.concat(parts, ignore_index=True)

# pred_long = to_long(pred, ID_COL, feat_cols, "pred_text")
# gt_long = to_long(gt, ID_COL, feat_cols, "gt_text")

# merged = pred_long.merge(gt_long, on=[ID_COL, "feature"], how="left", validate="m:1")

# def is_fail(s):
#     if s is None:
#         return True
#     if isinstance(s, float) and np.isnan(s):
#         return True
#     s = str(s)
#     return bool(FAIL_REGEX.match(s)) or (s.strip() == "")

# merged = merged[~merged["pred_text"].apply(is_fail)].copy()
# merged["gt_text"] = merged["gt_text"].fillna("").astype(str)
# merged = merged[merged["gt_text"].str.strip() != ""].copy()

# if LOWERCASE:
#     merged["pred_eval"] = merged["pred_text"].astype(str).str.strip().str.lower()
#     merged["gt_eval"] = merged["gt_text"].astype(str).str.strip().str.lower()
# else:
#     merged["pred_eval"] = merged["pred_text"].astype(str).str.strip()
#     merged["gt_eval"] = merged["gt_text"].astype(str).str.strip()

# if merged.empty:
#     print("Nothing to evaluate after filtering.")
#     pd.DataFrame(columns=[
#         "feature","n_pairs","bleu_corpus","rouge1_f1_mean","rougeL_f1_mean","berts_f1_mean"
#     ]).to_csv(os.path.join(OUT_DIR, "feature_summary.csv"), index=False)
#     raise SystemExit

# def corpus_bleu(preds, refs):
#     try:
#         return sacrebleu.corpus_bleu(preds, [refs]).score
#     except Exception:
#         return float("nan")

# def compute_bertscore(cands, refs, model, tokenizer, device, num_layers=9, batch_size=64):
#     """
#     Compute BERTScore F1 using local model
#     """
#     all_f1 = []
    
#     # Process in batches
#     for i in range(0, len(cands), batch_size):
#         batch_cands = cands[i:i+batch_size]
#         batch_refs = refs[i:i+batch_size]
        
#         # Tokenize
#         cand_inputs = tokenizer(batch_cands, padding=True, truncation=True, 
#                                 return_tensors='pt', max_length=512).to(device)
#         ref_inputs = tokenizer(batch_refs, padding=True, truncation=True, 
#                                return_tensors='pt', max_length=512).to(device)
        
#         with torch.no_grad():
#             # Get embeddings
#             cand_outputs = model(**cand_inputs, output_hidden_states=True)
#             ref_outputs = model(**ref_inputs, output_hidden_states=True)
            
#             # Use specified layer
#             cand_embs = cand_outputs.hidden_states[num_layers]
#             ref_embs = ref_outputs.hidden_states[num_layers]
            
#             # Compute cosine similarity for each pair
#             for j in range(len(batch_cands)):
#                 # Get valid tokens (non-padding)
#                 cand_mask = cand_inputs['attention_mask'][j].bool()
#                 ref_mask = ref_inputs['attention_mask'][j].bool()
                
#                 cand_emb = cand_embs[j][cand_mask]
#                 ref_emb = ref_embs[j][ref_mask]
                
#                 # Compute pairwise cosine similarity
#                 sim_matrix = torch.nn.functional.cosine_similarity(
#                     cand_emb.unsqueeze(1), ref_emb.unsqueeze(0), dim=2
#                 )
                
#                 # Greedy matching
#                 precision = sim_matrix.max(dim=1)[0].mean()
#                 recall = sim_matrix.max(dim=0)[0].mean()
                
#                 if precision + recall > 0:
#                     f1 = 2 * precision * recall / (precision + recall)
#                 else:
#                     f1 = torch.tensor(0.0)
                
#                 all_f1.append(f1.item())
    
#     return all_f1

# rscorer = rouge_scorer.RougeScorer(["rouge1","rougeL"], use_stemmer=True)

# rows = []
# for f, grp in merged.groupby("feature"):
#     preds = grp["pred_eval"].tolist()
#     refs = grp["gt_eval"].tolist()
#     n = len(grp)
    
#     r1, rL = [], []
#     for c, r in zip(preds, refs):
#         sc = rscorer.score(r, c)
#         r1.append(sc["rouge1"].fmeasure)
#         rL.append(sc["rougeL"].fmeasure)
    
#     print(f"Computing BERTScore for feature: {f} with {n} pairs...")
    
#     # Compute BERTScore using local model
#     f1_scores = compute_bertscore(preds, refs, model, tokenizer, device, NUM_LAYERS)
    
#     rows.append({
#         "feature": f,
#         "n_pairs": n,
#         "bleu_corpus": corpus_bleu(preds, refs),
#         "rouge1_f1_mean": float(np.mean(r1)),
#         "rougeL_f1_mean": float(np.mean(rL)),
#         "berts_f1_mean": float(np.mean(f1_scores)),
#     })

# feat_sum = pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)

# numeric_cols = ["n_pairs", "bleu_corpus", "rouge1_f1_mean", "rougeL_f1_mean", "berts_f1_mean"]
# avg_row = {"feature": "AVG"}
# avg_row["n_pairs"] = int(round(feat_sum["n_pairs"].mean()))
# for col in numeric_cols[1:]:
#     avg_row[col] = float(feat_sum[col].mean())

# feat_sum = pd.concat([feat_sum, pd.DataFrame([avg_row])], ignore_index=True)

# out_file = os.path.join(OUT_DIR, "Task2_InternVL3_5-8B_llmmerge_feature_summary.csv")
# feat_sum.to_csv(out_file, index=False)
# print(f"Saved -> {out_file}")



#previous
# # Download the bertscore model (run in terminal; replace <HF_CACHE_DIR> with your path):
# # huggingface-cli download microsoft/deberta-xlarge-mnli --cache-dir <HF_CACHE_DIR>

# import os, re
# import numpy as np
# import pandas as pd
# import sacrebleu
# from rouge_score import rouge_scorer
# from bert_score import score as bert_score

# os.environ["TRANSFORMERS_CACHE"] = "/mnt/SSD3/xinyi/hf_cache"

# PRED_CSV = "/mnt/SSD3/xinyi/benchmark/task2_metrics/Task2_llmmerge_results/Task1_InternVL3_5-8B_all_merged_llmmerge.csv"  
# GT_CSV   = "/mnt/SSD3/xinyi/benchmark/task2_metrics/task12_annotation.csv"   
# OUT_DIR  = "/mnt/SSD3/xinyi/benchmark/task2_metrics/Task2_llmmerge_results"
# ID_COL   = "file_name"

# LOWERCASE = True
# LANG = "en"                
# BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"    
# RESCALE_BERT = False
# FAIL_REGEX = re.compile(r"^\s*(fail(ed)?|error|n/?a|none|null)?\s*$", re.I)


# os.makedirs(OUT_DIR, exist_ok=True)
# pred = pd.read_csv(PRED_CSV)
# gt   = pd.read_csv(GT_CSV)


# J_PREFIX = "justification_for_"
# feat_cols = [c for c in pred.columns if c.startswith(J_PREFIX)]
# if not feat_cols:
#     raise ValueError(f"No columns start with '{J_PREFIX}' in prediction CSV.")

# def feat_name(col): return col[len(J_PREFIX):]

# def to_long(df, id_col, feat_cols, label):
#     parts = []
#     for c in feat_cols:
#         tmp = df[[id_col, c]].copy()
#         tmp.columns = [id_col, label]
#         tmp["feature"] = feat_name(c)
#         parts.append(tmp)
#     return pd.concat(parts, ignore_index=True)

# pred_long = to_long(pred, ID_COL, feat_cols, "pred_text")
# gt_long   = to_long(gt,   ID_COL, feat_cols, "gt_text")

# merged = pred_long.merge(gt_long, on=[ID_COL, "feature"], how="left", validate="m:1")

# def is_fail(s):
#     if s is None: return True
#     if isinstance(s, float) and np.isnan(s): return True
#     s = str(s)
#     return bool(FAIL_REGEX.match(s)) or (s.strip() == "")

# merged = merged[~merged["pred_text"].apply(is_fail)].copy()
# merged["gt_text"] = merged["gt_text"].fillna("").astype(str)
# merged = merged[merged["gt_text"].str.strip() != ""].copy()

# if LOWERCASE:
#     merged["pred_eval"] = merged["pred_text"].astype(str).str.strip().str.lower()
#     merged["gt_eval"]   = merged["gt_text"].astype(str).str.strip().str.lower()
# else:
#     merged["pred_eval"] = merged["pred_text"].astype(str).str.strip()
#     merged["gt_eval"]   = merged["gt_text"].astype(str).str.strip()

# if merged.empty:
#     print("Nothing to evaluate after filtering.")
#     pd.DataFrame(columns=[
#         "feature","n_pairs","bleu_corpus","rouge1_f1_mean","rougeL_f1_mean","berts_f1_mean"
#     ]).to_csv(os.path.join(OUT_DIR, "feature_summary.csv"), index=False)
#     raise SystemExit

# def corpus_bleu(preds, refs):
#     try: return sacrebleu.corpus_bleu(preds, [refs]).score
#     except Exception: return float("nan")

# rscorer = rouge_scorer.RougeScorer(["rouge1","rougeL"], use_stemmer=True)

# rows = []
# for f, grp in merged.groupby("feature"):
#     preds = grp["pred_eval"].tolist()
#     refs  = grp["gt_eval"].tolist()
#     n = len(grp)

#     r1, rL = [], []
#     for c, r in zip(preds, refs):
#         sc = rscorer.score(r, c)
#         r1.append(sc["rouge1"].fmeasure)
#         rL.append(sc["rougeL"].fmeasure)

#     _, _, F1 = bert_score(preds, refs, lang=LANG,
#                           model_type=BERTSCORE_MODEL,
#                           rescale_with_baseline=RESCALE_BERT)
#     rows.append({
#         "feature": f,
#         "n_pairs": n,
#         "bleu_corpus": corpus_bleu(preds, refs),
#         "rouge1_f1_mean": float(np.mean(r1)),
#         "rougeL_f1_mean": float(np.mean(rL)),
#         "berts_f1_mean": float(F1.numpy().mean()),
#     })

# feat_sum = pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)

# numeric_cols = ["n_pairs", "bleu_corpus", "rouge1_f1_mean", "rougeL_f1_mean", "berts_f1_mean"]

# avg_row = {"feature": "AVG"}
# avg_row["n_pairs"] = int(round(feat_sum["n_pairs"].mean()))

# for col in numeric_cols[1:]:
#     avg_row[col] = float(feat_sum[col].mean())

# feat_sum = pd.concat([feat_sum, pd.DataFrame([avg_row])], ignore_index=True)

# feat_sum.to_csv(os.path.join(OUT_DIR, "Task1_InternVL3_5-8B_llmmerge_feature_summary.csv"), index=False)
# print(f"Saved -> {os.path.join(OUT_DIR, 'Task1_InternVL3_5-8B_llmmerge_feature_summary.csv')}")

# # python /mnt/SSD3/xinyi/benchmark/task2_metrics/metrics.py
