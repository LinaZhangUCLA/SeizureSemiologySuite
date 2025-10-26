import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# ===================== CONFIG =====================
GT_CSV = "result/ground_truth/task12_annotation.csv"      
PRED_CSV = "result/vlm_inference/Qwen2.5-VL-7B-Instruct/Task6_Qwen2.5-VL-72B-Instruct_extracted_features.csv"     
GT_ID_COL = "file_name"      # GT使用 file_name
PRED_ID_COL = "video_name"   # Pred使用 video_name
OUT_CSV = "result/vlm_inference/Qwen2.5-VL-7B-Instruct/Task6_Qwen2.5-VL-72B_overall_precision_recall.csv"
# ==================================================

# 读取数据
gt = pd.read_csv(GT_CSV)
pred = pd.read_csv(PRED_CSV)

print(f"[Info] GT shape: {gt.shape}, Pred shape: {pred.shape}")

# 检查ID列是否存在
if GT_ID_COL not in gt.columns:
    raise ValueError(f"GT CSV缺少列: {GT_ID_COL}. 实际列: {gt.columns.tolist()}")
if PRED_ID_COL not in pred.columns:
    raise ValueError(f"Pred CSV缺少列: {PRED_ID_COL}. 实际列: {pred.columns.tolist()}")

# 统一ID列名为 'id'
gt = gt.rename(columns={GT_ID_COL: 'id'})
pred = pred.rename(columns={PRED_ID_COL: 'id'})
ID_COL = 'id'

# 排序并重置索引
gt = gt.sort_values(ID_COL).reset_index(drop=True)
pred = pred.sort_values(ID_COL).reset_index(drop=True)

# 检查ID是否匹配
if not gt[ID_COL].equals(pred[ID_COL]):
    print("[WARNING] GT和Pred的video_name不完全匹配!")
    print(f"GT有 {len(gt)} 行, Pred有 {len(pred)} 行")
    
    # 取交集
    common_ids = set(gt[ID_COL]) & set(pred[ID_COL])
    print(f"共同的video_name数量: {len(common_ids)}")
    
    gt = gt[gt[ID_COL].isin(common_ids)].sort_values(ID_COL).reset_index(drop=True)
    pred = pred[pred[ID_COL].isin(common_ids)].sort_values(ID_COL).reset_index(drop=True)
    print(f"[Info] 使用交集后: GT shape: {gt.shape}, Pred shape: {pred.shape}")

# 找出症状列 (排除ID列和包含justification/time的列)
exclude_cols = [ID_COL] + [c for c in gt.columns if "justification" in c.lower() or "time" in c.lower()]
symptom_cols = [c for c in gt.columns if c not in exclude_cols]

print(f"[Info] 症状列数量: {len(symptom_cols)}")
print(f"[Info] 症状列: {symptom_cols}")

# 检查pred是否包含所有症状列
missing_cols = [c for c in symptom_cols if c not in pred.columns]
if missing_cols:
    raise ValueError(f"Pred CSV缺少以下症状列: {missing_cols}")

# 转换为二进制
# GT: "yes" -> 1, 其他 -> 0
gt_bin = gt[symptom_cols].map(lambda x: 1 if str(x).strip().lower() == "yes" else 0)

# Pred: 1 -> 1, 0 -> 0, "NA"或其他 -> 0
def convert_pred(x):
    try:
        val = int(x)
        return 1 if val == 1 else 0
    except (ValueError, TypeError):
        return 0  # 将 "NA" 等非数字值当作 0

pred_bin = pred[symptom_cols].map(convert_pred)

# 计算 micro-averaged 指标
micro_prec = precision_score(gt_bin.values.ravel(), pred_bin.values.ravel(), zero_division=0)
micro_rec = recall_score(gt_bin.values.ravel(), pred_bin.values.ravel(), zero_division=0)
micro_f1 = f1_score(gt_bin.values.ravel(), pred_bin.values.ravel(), zero_division=0)

# 保存结果
results = pd.DataFrame([{
    "precision": micro_prec,
    "recall": micro_rec,
    "f1_score": micro_f1
}])
results.to_csv(OUT_CSV, index=False)

print(f"\n[Done] Saved overall results -> {OUT_CSV}")


#previous
# import pandas as pd
# from sklearn.metrics import precision_score, recall_score, f1_score

# GT_CSV = "/Users/pxy_amber/Downloads/task2_metrics/task12_annotation.csv"      
# PRED_CSV = "/Users/pxy_amber/Downloads/task6_metrics/Task6_llmmerge/Task6_Qwen2.5-VL-72B-Instruct_extracted_features_qwen_plus.csv"     
# ID_COL = "file_name"             
# OUT_CSV = "/Users/pxy_amber/Downloads/task6_metrics/Task6_Qwen2.5-VL-72B_overall_precision_recall.csv"

# gt = pd.read_csv(GT_CSV)
# pred = pd.read_csv(PRED_CSV)

# gt = gt.sort_values(ID_COL).reset_index(drop=True)
# pred = pred.sort_values(ID_COL).reset_index(drop=True)

# exclude_cols = [ID_COL] + [c for c in gt.columns if "justification" in c or "time" in c]
# symptom_cols = [c for c in gt.columns if c not in exclude_cols]

# gt_bin = gt[symptom_cols].applymap(lambda x: 1 if str(x).lower() == "yes" else 0)
# pred_bin = pred[symptom_cols].astype(int)

# micro_prec = precision_score(gt_bin.values.ravel(), pred_bin.values.ravel(), zero_division=0)
# micro_rec = recall_score(gt_bin.values.ravel(), pred_bin.values.ravel(), zero_division=0)
# micro_f1 = f1_score(gt_bin.values.ravel(), pred_bin.values.ravel(), zero_division=0)

# results = pd.DataFrame([{
#     "precision": micro_prec,
#     "recall": micro_rec,
#     "f1_score": micro_f1
# }])
# results.to_csv(OUT_CSV, index=False)

# print(f"Saved overall results -> {OUT_CSV}")
# print(results)
