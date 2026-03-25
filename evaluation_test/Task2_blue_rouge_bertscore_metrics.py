# Download the bertscore model (run in terminal; replace <HF_CACHE_DIR> with your path):
# huggingface-cli download microsoft/deberta-xlarge-mnli --cache-dir <HF_CACHE_DIR>

import argparse
import os, re, glob
import numpy as np
import pandas as pd
import sacrebleu
from rouge_score import rouge_scorer
import torch
from transformers import AutoTokenizer, AutoModel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VLM_INFERENCE_DIR = os.path.join(REPO_ROOT, "result", "vlm_inference_test")
GT_CSV = os.path.join(REPO_ROOT, "result", "ground_truth", "task12_annotation.csv")
OUT_DIR = os.path.join(REPO_ROOT, "metrics_test", "Task2_feature_metrics")
ID_COL = "file_name"
LOWERCASE = True
LANG = "en"

# Local model path
#LOCAL_MODEL_PATH = '/Users/eehan/Desktop/deberta-xlarge-mnli'
LOCAL_MODEL_PATH = "/mnt/SSD3/lina/bertmodel"
# Number of layers to use (DeBERTa-xlarge has 24 layers, typically use layer 9 for similarity)
NUM_LAYERS = 9

FAIL_REGEX = re.compile(r"^\s*(fail(ed)?|error|n/?a|none|null)?\s*$", re.I)

MODELS = [
    "InternVL3_5-8B",
    "InternVL3_5-38B",
    "Qwen2.5-VL-7B-Instruct",
    "Qwen2.5-VL-32B-Instruct",
    "Qwen2.5-VL-72B-Instruct",
    "Lingshu-32B",
    "Qwen2.5-Omni-7B",
    'Qwen3-VL-8B-Instruct',
    'Qwen3-VL-32B-Instruct',
    "Qwen3-Omni-30B-A3B-Instruct",
    'seizure_omni_sft',
    'seizure_omni_grpo'
]    


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


def normalize_model_name(name):
    """标准化模型名称，用于匹配（忽略大小写、分隔符）"""
    return name.lower().replace('-', '').replace('_', '').replace('.', '')


def process_csv(pred_csv_path, gt, model, tokenizer, device, out_dir):
    """Process a single prediction CSV file"""
    pred_filename = os.path.basename(pred_csv_path)
    
    # 修复：使用非贪婪匹配，并在 Instruct 或下划线处停止
    match = re.search(r'(Qwen[\w\.-]+?|InternVL[\w\.-]+?)(?:-Instruct|_all_merged)', pred_filename, re.IGNORECASE)
    if match:
        model_identifier = match.group(1)
    else:
        folder_name = os.path.basename(os.path.dirname(pred_csv_path))
        model_identifier = folder_name

    # 清理标识符（移除可能残留的 -Instruct）
    model_identifier = re.sub(r'-Instruct.*$', '', model_identifier, flags=re.IGNORECASE)

    # 生成输出文件名
    output_filename = f"{model_identifier}_feature_summary.csv"
    out_file = os.path.join(out_dir, output_filename)

    print(f"Processing: {pred_csv_path}")
    print(f"  Model identifier: {model_identifier}")
    print(f"  Output file: {out_file}")

    # 检查输出文件是否已存在（精确匹配）
    if os.path.exists(out_file):
        print(f"  ✓ Output file already exists, skipping.\n")
        return

    # 检查是否存在相似的文件（标准化后匹配）
    existing_files = glob.glob(os.path.join(OUT_DIR, "*_feature_summary.csv"))
    normalized_target = normalize_model_name(model_identifier)
    
    for existing in existing_files:
        existing_model = os.path.basename(existing).replace('_feature_summary.csv', '')
        normalized_existing = normalize_model_name(existing_model)
        
        if normalized_target == normalized_existing:
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


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Task 2 NLP metrics.")
    parser.add_argument("--pred_csv", default=None, help="Optional single Task 2 llm-merged CSV.")
    parser.add_argument("--base_dir", default=VLM_INFERENCE_DIR, help="Base directory containing per-model result folders.")
    parser.add_argument("--models", nargs="*", default=MODELS, help="Model folder names for batch mode.")
    parser.add_argument("--gt_csv", default=GT_CSV, help="Ground-truth Task 1/2 annotation CSV.")
    parser.add_argument("--out_dir", default=OUT_DIR, help="Output directory for per-feature summary CSVs.")
    parser.add_argument("--local_model_path", default=LOCAL_MODEL_PATH, help="Local DeBERTa model path for BERTScore.")
    parser.add_argument("--device", default="cuda:2" if torch.cuda.is_available() else "cpu",
                        help="Torch device for similarity model, e.g. cuda:0 or cpu.")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.pred_csv:
        pred_csv_files = [args.pred_csv]
    else:
        pred_csv_files = [
            os.path.join(args.base_dir, model, f"Task12_{model}_all_merged_llmmerge.csv")
            for model in args.models
        ]

    print(f"Found {len(pred_csv_files)} CSV file(s) to process:")
    for f in pred_csv_files:
        print(f"  - {f}")
    print()

    print(f"Loading model from: {args.local_model_path}")
    if not os.path.exists(args.local_model_path):
        raise FileNotFoundError(f"Model directory not found: {args.local_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.local_model_path)
    model = AutoModel.from_pretrained(args.local_model_path)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully on {device}!\n")

    gt = pd.read_csv(args.gt_csv)

    print("=" * 60)
    print("Starting batch processing...")
    print("=" * 60 + "\n")

    for pred_csv in pred_csv_files:
        try:
            process_csv(pred_csv, gt, model, tokenizer, device, args.out_dir)
        except Exception as e:
            print(f"✗ Error processing {pred_csv}: {e}\n")
            continue

    print("=" * 60)
    print("Batch processing complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
