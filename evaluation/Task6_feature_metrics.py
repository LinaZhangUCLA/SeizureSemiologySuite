import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# ===================== Configuration (paths hidden) =====================
# Prefer environment variables to avoid hardcoding. Replace <BASE_DIR> if not using env.
BASE_DIR = os.environ.get("PROJECT_BASE_DIR", "<BASE_DIR>")

# Ground-truth CSV shared by all models
GT_CSV = os.path.join(BASE_DIR, "task2_metrics", "task12_annotation.csv")

# Input/Output filename patterns (must contain {model})
# Example input : <BASE_DIR>/task6_metrics/Task6_llmmerge/Task6_{model}_extracted_features_qwen.csv
# Example output: <BASE_DIR>/task6_metrics/Task6_metrics/Task6_{model}_overall_precision_recall.csv
PRED_PATTERN = os.path.join(BASE_DIR, "task6_metrics", "Task6_llmmerge", "Task6_{model}_extracted_features_qwen.csv")
OUT_DIR      = os.path.join(BASE_DIR, "task6_metrics", "Task6_metrics")

# Primary key column
ID_COL = "file_name"

# Models to evaluate (AF3 excluded)
MODELS = [
    "InternVL3_5-8B",
    "InternVL3_5-38B",
    "Qwen2.5-VL-7B-Instruct",
    "Qwen2.5-VL-32B-Instruct",
    "Qwen2.5-VL-72B-Instruct",
    "Lingshu-32B",
]
# =======================================================================


def to_01_nan(df_sub: pd.DataFrame) -> pd.DataFrame:
    """
    Map '0/1/na' (case-insensitive) to numeric with NaN for NA.
    Any non-0/1 parses to NaN.
    """
    tmp = df_sub.astype(str).applymap(lambda s: s.strip().lower())
    tmp = tmp.replace({"na": np.nan, "n/a": np.nan, "nan": np.nan, "": np.nan})
    tmp = tmp.apply(pd.to_numeric, errors="coerce")  # expect 0/1, else NaN
    return tmp.astype("float64")


def select_symptom_cols(gt_df: pd.DataFrame) -> list:
    """
    Use all GT columns except ID/justification/time-related columns.
    """
    exclude = [ID_COL] + [c for c in gt_df.columns if "justification" in c or "time" in c]
    return [c for c in gt_df.columns if c not in exclude]


def align_by_id(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align GT and prediction frames by ID; fall back to inner join if needed.
    """
    gt_df = gt_df.sort_values(ID_COL).reset_index(drop=True)
    pred_df = pred_df.sort_values(ID_COL).reset_index(drop=True)
    if not gt_df[ID_COL].equals(pred_df[ID_COL]):
        ids = gt_df[[ID_COL]].merge(pred_df[[ID_COL]], on=ID_COL, how="inner")
        gt_df = gt_df.merge(ids, on=ID_COL, how="inner").sort_values(ID_COL).reset_index(drop=True)
        pred_df = pred_df.merge(ids, on=ID_COL, how="inner").sort_values(ID_COL).reset_index(drop=True)
    return gt_df, pred_df


def evaluate_one_model(model_name: str, gt_csv: str, pred_csv: str, out_dir: str) -> pd.DataFrame:
    """
    Compute micro precision/recall/F1 for one model, skipping NA cells on either side.
    Returns a single-row DataFrame with metrics and metadata.
    """
    # Load
    gt = pd.read_csv(gt_csv)
    try:
        pred = pd.read_csv(pred_csv)
    except FileNotFoundError:
        print(f"[WARN] Missing prediction CSV for {model_name}: {pred_csv}")
        return pd.DataFrame()

    # Schema check
    if ID_COL not in gt.columns or ID_COL not in pred.columns:
        print(f"[WARN] {model_name}: '{ID_COL}' column missing.")
        return pd.DataFrame()

    # Symptom columns from GT
    symptom_cols = select_symptom_cols(gt)
    missing_preds = [c for c in symptom_cols if c not in pred.columns]
    if missing_preds:
        print(f"[WARN] {model_name}: prediction missing columns: {missing_preds}")
        return pd.DataFrame()

    # Align by ID
    gt, pred = align_by_id(gt, pred)

    # Map to numeric with NaN for NA
    gt_num   = to_01_nan(gt[symptom_cols])
    pred_num = to_01_nan(pred[symptom_cols])

    # Valid mask: both sides present (0/1)
    mask = (~gt_num.isna()) & (~pred_num.isna())
    n_valid = int(mask.values.sum())

    if n_valid == 0:
        row = {
            "model": model_name,
            "precision": np.nan,
            "recall": np.nan,
            "f1_score": np.nan,
            "n_valid_pairs": 0,
        }
    else:
        y_true = gt_num[mask].values.ravel().astype(int)
        y_pred = pred_num[mask].values.ravel().astype(int)
        row = {
            "model": model_name,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
            "f1_score":  f1_score(y_true, y_pred, zero_division=0),
            "n_valid_pairs": n_valid,
        }

    # Save per-model CSV
    os.makedirs(out_dir, exist_ok=True)
    per_model_path = os.path.join(out_dir, f"Task6_{model_name}_overall_precision_recall.csv")
    pd.DataFrame([row]).drop(columns=["model"]).to_csv(per_model_path, index=False)
    print(f"[OK] {model_name} -> {per_model_path}")

    return pd.DataFrame([row])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Run all models in sequence
    all_rows = []
    for m in MODELS:
        pred_csv = PRED_PATTERN.format(model=m)
        df_row = evaluate_one_model(m, GT_CSV, pred_csv, OUT_DIR)
        if not df_row.empty:
            all_rows.append(df_row)

    # Save overall table
    if all_rows:
        overall = pd.concat(all_rows, ignore_index=True)
        overall_path = os.path.join(OUT_DIR, "Task6_overall_precision_recall_all_models.csv")
        overall.to_csv(overall_path, index=False)
        print(f"[OK] Overall -> {overall_path}")
        print(overall)
    else:
        print("[INFO] No results produced (check input files and columns).")


if __name__ == "__main__":
    main()
