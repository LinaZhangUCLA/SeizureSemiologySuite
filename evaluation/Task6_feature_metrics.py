import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

GT_CSV = "/Users/pxy_amber/Downloads/task2_metrics/task12_annotation.csv"      
PRED_CSV = "/Users/pxy_amber/Downloads/task6_metrics/Task6_llmmerge/Task6_Qwen2.5-VL-72B-Instruct_extracted_features_qwen_plus.csv"     
ID_COL = "file_name"             
OUT_CSV = "/Users/pxy_amber/Downloads/task6_metrics/Task6_Qwen2.5-VL-72B_overall_precision_recall.csv"

gt = pd.read_csv(GT_CSV)
pred = pd.read_csv(PRED_CSV)

gt = gt.sort_values(ID_COL).reset_index(drop=True)
pred = pred.sort_values(ID_COL).reset_index(drop=True)

exclude_cols = [ID_COL] + [c for c in gt.columns if "justification" in c or "time" in c]
symptom_cols = [c for c in gt.columns if c not in exclude_cols]

gt_bin = gt[symptom_cols].applymap(lambda x: 1 if str(x).lower() == "yes" else 0)
pred_bin = pred[symptom_cols].astype(int)

micro_prec = precision_score(gt_bin.values.ravel(), pred_bin.values.ravel(), zero_division=0)
micro_rec = recall_score(gt_bin.values.ravel(), pred_bin.values.ravel(), zero_division=0)
micro_f1 = f1_score(gt_bin.values.ravel(), pred_bin.values.ravel(), zero_division=0)

results = pd.DataFrame([{
    "precision": micro_prec,
    "recall": micro_rec,
    "f1_score": micro_f1
}])
results.to_csv(OUT_CSV, index=False)

print(f"Saved overall results -> {OUT_CSV}")
print(results)
