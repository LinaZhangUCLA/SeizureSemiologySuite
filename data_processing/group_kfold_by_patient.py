#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group K-Fold split by patient ID (from filename before the first '@').

Usage (example):
    python group_kfold_by_patient.py \
        -i Task7_annotation.csv \
        -o cv_folds.csv \
        -c file_name \
        -k 5 \
        --pairs-dir cv_splits

Notes:
- Requires: pandas, scikit-learn, numpy
  Install: pip install pandas scikit-learn numpy
- Patient ID is parsed as everything BEFORE the first '@' in the basename of the file_name.
  Example: 'A0002@5-13-2021@UA6693LK@sz_v1_1.mp4' -> patient_id = 'A0002'
- Output 1 (always): a CSV with columns: [original columns..., patient_id, cv_fold]
  where cv_fold in {0, 1, 2, 3, 4}. For fold k, use rows where cv_fold == k as validation,
  and the rest as training.
- Output 2 (optional): if --pairs-dir is provided, writes per-fold train/val CSVs:
  <pairs-dir>/fold{k}_train.csv and <pairs-dir>/fold{k}_val.csv for k in 0..k-1.
"""

import argparse
import os
import sys
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def extract_patient_id(path: str) -> str:
    """
    Extract patient ID from a file path by taking the basename and
    returning the substring before the first '@'. If '@' is not present,
    this returns the whole stem (without extension).
    """
    base = os.path.basename(str(path))
    if '@' in base:
        return base.split('@', 1)[0]
    # Fallback: use stem without extension
    stem = os.path.splitext(base)[0]
    # If there's an '@' in the stem (rare), still split
    if '@' in stem:
        return stem.split('@', 1)[0]
    return stem


def compute_group_kfold(df: pd.DataFrame, filename_col: str, n_splits: int) -> pd.DataFrame:
    """
    Add 'patient_id' and 'cv_fold' columns to df using GroupKFold on patient_id.
    """
    if filename_col not in df.columns:
        raise KeyError(
            f"Column '{filename_col}' not found in CSV. Available columns: {list(df.columns)}"
        )

    # Derive patient_id
    patient_ids = df[filename_col].astype(str).map(extract_patient_id)
    df = df.copy()
    df["patient_id"] = patient_ids

    # Prepare group k-fold
    groups = df["patient_id"].values
    gkf = GroupKFold(n_splits=n_splits)

    # We'll assign each sample to exactly one validation fold
    cv_fold = np.full(len(df), -1, dtype=int)

    # Use dummy X, y since we only care about groups
    X_dummy = np.zeros((len(df), 1))
    y_dummy = None

    for fold_id, (_, val_idx) in enumerate(gkf.split(X_dummy, y_dummy, groups=groups)):
        cv_fold[val_idx] = fold_id

    if (cv_fold < 0).any():
        raise RuntimeError("Some samples were not assigned to a fold (cv_fold == -1).")

    df["cv_fold"] = cv_fold

    # Sanity check: no patient appears in multiple validation folds
    # (equivalently, each patient_id maps to a unique cv_fold among its rows)
    duplicates = (
        df.groupby("patient_id")["cv_fold"]
        .nunique()
        .reset_index(name="n_unique_folds")
    )
    if (duplicates["n_unique_folds"] > 1).any():
        bad = duplicates[duplicates["n_unique_folds"] > 1]["patient_id"].tolist()
        raise RuntimeError(
            f"Group leakage detected: the following patient_id(s) appear in multiple folds: {bad}"
        )

    return df


def write_pairs(df: pd.DataFrame, pairs_dir: str, n_splits: int, filename_col: str) -> None:
    """
    Write per-fold train/val CSV pairs into pairs_dir.
    """
    os.makedirs(pairs_dir, exist_ok=True)
    for k in range(n_splits):
        val_df = df[df["cv_fold"] == k].copy()
        train_df = df[df["cv_fold"] != k].copy()
        val_path = os.path.join(pairs_dir, f"fold{k}_val.csv")
        train_path = os.path.join(pairs_dir, f"fold{k}_train.csv")
        val_df.to_csv(val_path, index=False)
        train_df.to_csv(train_path, index=False)


def print_stats(df: pd.DataFrame, n_splits: int) -> None:
    """
    Print useful stats about fold balance (videos and unique patients per fold).
    """
    total_videos = len(df)
    total_patients = df["patient_id"].nunique()
    print(f"Total videos: {total_videos}")
    print(f"Total unique patients: {total_patients}")
    print("-" * 60)
    for k in range(n_splits):
        val_df = df[df["cv_fold"] == k]
        train_df = df[df["cv_fold"] != k]
        print(f"Fold {k}:")
        print(f"  Val -> videos: {len(val_df):5d}, patients: {val_df['patient_id'].nunique():5d}")
        print(f"  Train -> videos: {len(train_df):5d}, patients: {train_df['patient_id'].nunique():5d}")
    print("-" * 60)

    # Quick leakage check per fold
    for k in range(n_splits):
        val_patients = set(df[df["cv_fold"] == k]["patient_id"])
        train_patients = set(df[df["cv_fold"] != k]["patient_id"])
        leakage = val_patients.intersection(train_patients)
        assert len(leakage) == 0, f"Leakage detected in fold {k}: {leakage}"


def main():
    parser = argparse.ArgumentParser(description="5-fold GroupKFold split by patient ID extracted from file_name.")
    parser.add_argument("-i", "--input", required=True, help="Input CSV path (must contain the filename column).")
    parser.add_argument("-o", "--output", default="cv_folds.csv", help="Output CSV path to save with 'patient_id' and 'cv_fold'.")
    parser.add_argument("-c", "--column", default="file_name", help="Name of the column containing video file names (default: file_name).")
    parser.add_argument("-k", "--folds", type=int, default=5, help="Number of folds (default: 5).")
    parser.add_argument("--pairs-dir", default=None, help="If set, write per-fold train/val CSVs into this directory.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"ERROR: Failed to read CSV '{args.input}': {e}", file=sys.stderr)
        sys.exit(1)

    try:
        df_out = compute_group_kfold(df, filename_col=args.column, n_splits=args.folds)
    except Exception as e:
        print(f"ERROR during fold computation: {e}", file=sys.stderr)
        sys.exit(1)

    # Save the main output CSV
    try:
        df_out.to_csv(args.output, index=False)
    except Exception as e:
        print(f"ERROR: Failed to write output CSV '{args.output}': {e}", file=sys.stderr)
        sys.exit(1)

    # Optionally, save per-fold train/val pairs
    if args.pairs_dir:
        try:
            write_pairs(df_out, args.pairs_dir, n_splits=args.folds, filename_col=args.column)
        except Exception as e:
            print(f"ERROR: Failed to write per-fold pairs in '{args.pairs_dir}': {e}", file=sys.stderr)
            sys.exit(1)

    # Print stats and a success message
    print_stats(df_out, n_splits=args.folds)
    print(f"\nSaved main split file to: {args.output}")
    if args.pairs_dir:
        print(f"Saved per-fold train/val CSVs under: {args.pairs_dir}")


if __name__ == "__main__":
    main()
