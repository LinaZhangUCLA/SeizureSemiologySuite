#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_patient_group_stratified_5fold.py

Description:
- For the scenario of "438 videos, labels include ES / NES / Sample (only 6 Sample)":
  * Automatically ignore samples with label == 'Sample' (case-insensitive);
  * Extract patient ID from file_name (substring before the first '@');
  * Perform group-wise (by patient) 5-fold stratified splitting (preserve ES/NES ratio overall);
  * Export per-fold train/val CSVs, a combined CSV, ignored samples list, and fold statistics.

Input CSV must contain at least:
- file_name: video filename, e.g. A0002@5-13-2021@UA6693LK@sz_v1_1.mp4
- label: label, e.g. ES / NES / Sample (case-insensitive)

Outputs:
- out_dir/all_folds.csv: samples (ES/NES only) with assigned fold (0..K-1 for validation fold)
- out_dir/fold{i}_train.csv, out_dir/fold{i}_val.csv: per-fold train/val splits (ES/NES only)
- out_dir/ignored_samples.csv: list of ignored Sample rows
- out_dir/fold_stats.csv: summary of ES/NES counts and ratios per fold

Dependencies:
- pandas, numpy
- scikit-learn (if StratifiedGroupKFold is available it will be used; otherwise the script falls back to a built-in heuristic)

Usage:
python make_patient_group_stratified_5fold.py --csv data.csv --outdir out_dir \
  --filename-col file_name --label-col label --n-splits 5 --seed 100
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Tuple
import numpy as np
import pandas as pd


def extract_patient_id(file_name: str) -> str:
    if not isinstance(file_name, str):
        file_name = str(file_name)
    return file_name.split("@", 1)[0] if "@" in file_name else file_name


def filter_es_nes(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lab_upper = df[label_col].astype(str).str.strip().str.upper()
    is_sample = lab_upper == "SAMPLE"
    ignored = df[is_sample].copy()
    kept = df[~is_sample].copy()

    kept_upper = kept[label_col].astype(str).str.strip().str.upper()
    bad = ~kept_upper.isin(["ES", "NES"])
    if bad.any():
        raise ValueError(f"Detected labels other than ES/NES/Sample:{sorted(kept[label_col][bad].unique())}")
    kept[label_col] = kept_upper
    ignored[label_col] = ignored[label_col].astype(str).str.strip()
    return kept, ignored


def try_stratified_group_kfold(n_splits: int, seed: int, y, groups):
    try:
        from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        idx = np.arange(len(y))
        return sgkf.split(idx, y, groups)
    except Exception:
        return None


def greedy_group_stratified_split(
    y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int
):
    rng = np.random.RandomState(seed)
    n = len(y)
    assert n == len(groups)

    df = pd.DataFrame({"group": groups, "y": y})
    gstat = df.groupby("group")["y"].agg(["count", "sum"]).rename(columns={"count": "n", "sum": "n_pos"})
    gstat["imbalance"] = (2 * gstat["n_pos"] - gstat["n"]).abs()
    gstat["rand"] = rng.rand(len(gstat))
    gstat = gstat.sort_values(by=["imbalance", "n", "rand"], ascending=[False, False, True])

    total_n = n
    total_pos = int(y.sum())
    global_ratio = total_pos / total_n if total_n > 0 else 0.0
    target_size = total_n / n_splits
    lam = 0.1

    fold_n = np.zeros(n_splits, dtype=int)
    fold_pos = np.zeros(n_splits, dtype=int)

    group_to_fold: Dict[str, int] = {}

    for g, row in gstat.iterrows():
        g_n = int(row["n"])
        g_pos = int(row["n_pos"])
        best_fold = None
        best_score = None

        for f in range(n_splits):
            n_after = fold_n[f] + g_n
            pos_after = fold_pos[f] + g_pos
            ratio_after = (pos_after / n_after) if n_after > 0 else 0.0
            score = abs(ratio_after - global_ratio) + lam * (abs(n_after - target_size) / max(1.0, target_size))

            if (best_score is None) or (score < best_score):
                best_score = score
                best_fold = f

        group_to_fold[g] = int(best_fold)
        fold_n[best_fold] += g_n
        fold_pos[best_fold] += g_pos

    fold_idx = np.array([group_to_fold[g] for g in groups], dtype=int)
    return fold_idx


def summarize_folds(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Generate summary table of total and per-fold sample counts and label distributions (ES/NES only)."""
    def label_counts(frame: pd.DataFrame) -> pd.Series:
        cnt = frame[label_col].astype(str).str.upper().value_counts()
        es = int(cnt.get("ES", 0))
        nes = int(cnt.get("NES", 0))
        total = es + nes
        ratio_es = (es / total) if total > 0 else 0.0
        return pd.Series({"n_total": total, "n_ES": es, "n_NES": nes, "ratio_ES": ratio_es})

    rows = []
    overall = label_counts(df)
    overall.name = "ALL"
    rows.append(overall)

    for f, dsub in df.groupby("fold"):
        row = label_counts(dsub)
        row.name = f"fold_{int(f)}"
        rows.append(row)

    return pd.DataFrame(rows)


def main():

    parser = argparse.ArgumentParser(description="Group-aware 5-fold stratified split by patient ID; ignore 'Sample' and preserve ES/NES ratio.")
    parser.add_argument("--csv", type=str, default="./../result/ground_truth/task7_annotation.csv",help="Input CSV path; must contain file_name and label columns")
    parser.add_argument("--outdir", type=str,default = "datasplit",help="Output directory")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of folds (default=5)")
    parser.add_argument("--seed", type=int, default=100, help="Random seed (default=100)")
    parser.add_argument("--filename-col", default="file_name", help="Filename column name (default=file_name)")
    parser.add_argument("--label-col", default="label", help="Label column name (default=label)")
    args = parser.parse_args()


    os.makedirs(args.outdir, exist_ok=True)


    df = pd.read_csv(args.csv)
    if args.filename_col not in df.columns:
        raise KeyError(f"Column not found:{args.filename_col}")
    if args.label_col not in df.columns:
        raise KeyError(f"Column not found:{args.label_col}")

    # Ignore 'Sample', keep only ES/NES
    kept, ignored = filter_es_nes(df, args.label_col)

    # Save ignored samples
    ignored_path = os.path.join(args.outdir, "ignored_samples.csv")
    ignored.to_csv(ignored_path, index=False, encoding="utf-8")
    print(f"[INFO] Ignored Sample count: {len(ignored)} (saved: {ignored_path})")

    # Extract patient_id
    kept = kept.copy()
    kept["patient_id"] = kept[args.filename_col].map(extract_patient_id)

    # Map labels to binary (ES=1, NES=0) for stratification
    y = (kept[args.label_col].str.upper() == "ES").astype(int).values
    groups = kept["patient_id"].astype(str).values

    # Basic checks
    n_groups = kept["patient_id"].nunique()
    if n_groups < args.n_splits:
        raise ValueError(f"Number of available patients ({n_groups}) is less than n_splits ({args.n_splits}); cannot do grouped K-fold.")

    # Prefer sklearn's StratifiedGroupKFold if available
    split_gen = try_stratified_group_kfold(args.n_splits, args.seed, y, groups)
    if split_gen is not None:
        fold_of_idx = np.full(len(kept), -1, dtype=int)
        for fold_id, (_, val_idx) in enumerate(split_gen):
            fold_of_idx[val_idx] = fold_id
        assert (fold_of_idx >= 0).all(), "There are samples with no assigned fold"
    else:
        # Fallback to greedy heuristic
        print("***** use greedy_group_stratified_split *****")
        fold_of_idx = greedy_group_stratified_split(y, groups, n_splits=args.n_splits, seed=args.seed)


    kept = kept.copy()
    kept["fold"] = fold_of_idx


    #print all patients in each fold
    # for f in sorted(kept["fold"].unique()):
    #     f = int(f)
    #     val_df = kept[kept["fold"] == f].copy()
    #     train_df = kept[kept["fold"] != f].copy()
    #     val_patients = set(val_df["patient_id"].unique())
    #     train_patients = set(train_df["patient_id"].unique())
    #     print(f"Fold {f}:")
    #     print(f"  Val patients ({len(val_patients)}): {sorted(val_patients)}")
    #     print(f"  Train patients ({len(train_patients)}): {sorted(train_patients)}")

    #     check_overlap = val_patients.intersection(train_patients)
    #     print(f"  Overlap patients: {sorted(check_overlap)} for fold {f}")

    # Save combined file
    all_path = os.path.join(args.outdir, "all_folds.csv")
    kept.to_csv(all_path, index=False, encoding="utf-8")
    print(f"[保存] {all_path}")

    # Export per-fold train/val
    for f in sorted(kept["fold"].unique()):
        f = int(f)
        val_df = kept[kept["fold"] == f].copy()
        train_df = kept[kept["fold"] != f].copy()
        val_path = os.path.join(args.outdir, f"fold{f}_val.csv")
        train_path = os.path.join(args.outdir, f"fold{f}_train.csv")
        val_df.to_csv(val_path, index=False, encoding="utf-8")
        train_df.to_csv(train_path, index=False, encoding="utf-8")
        print(f"[SAVED] fold {f}: {train_path} / {val_path}")

    # Statistics
    stat = summarize_folds(kept, label_col=args.label_col)
    stat_path = os.path.join(args.outdir, "fold_stats.csv")
    stat.to_csv(stat_path, encoding="utf-8")
    print(f"[SAVED] {stat_path}")
    print("\n=== Stats preview (ES/NES only) ===")
    print(stat)


if __name__ == "__main__":
    main()
