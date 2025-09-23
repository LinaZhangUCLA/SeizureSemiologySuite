
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ES vs NES Percentage Plotter
----------------------------
- Inputs: two CSVs (ES-only rows, NES-only rows) containing 0/1 feature flags.
- Output: a grouped bar chart (%) and a CSV table of ES_% / NES_%.
Usage:
    python es_nes_percentage_plot.py             --es_csv feature_count_with_flags_ES_only.csv             --nes_csv feature_count_with_flags_NES_only.csv             --out_prefix es_nes_percent

Notes:
- Normalization is within group: ES columns divided by #ES rows; NES columns divided by #NES rows.
- Column name fixes: "arm_flex" -> "arm_flexion", "verbar" -> "verbal_responsiveness".
- Order is enforced by the list below; missing features (if any) are appended at the end.
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List

ORDER = ["ictal_vocalization", "arm_flexion", "tonic", "clonic",
         "head_turning", "arm_straightening", "eye_blinking", "full_body_shaking",
         "face_pulling", "occur_during_sleep", "figure4", "oral_automatisms",
         "face_twitching", "limb_automatisms", "blank_stare", "arms_move_simultaneously",
         "verbal_responsiveness", "close_eyes", "asynchronous_movement", "pelvic_thrusting"]

RENAME_MAP = {
    "arm_flex": "arm_flexion",
    "verbar": "verbal_responsiveness"
}

DROP_COLS = ["Unnamed: 0", "file_name", "es_flag", "nes_flag"]

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop known non-feature columns if they exist
    keep_cols = [c for c in df.columns if c not in DROP_COLS]
    df = df[keep_cols].copy()
    # Rename to canonical feature names
    df = df.rename(columns=RENAME_MAP)
    # Ensure only 0/1 numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df

def normalize_percentage(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    if n == 0:
        return pd.Series({c: 0.0 for c in df.columns})
    return (df.sum() / n * 100.0)

def align_columns(es: pd.DataFrame, nes: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, List[str]):
    all_cols = sorted(set(es.columns).union(set(nes.columns)))
    es2 = es.reindex(columns=all_cols, fill_value=0)
    nes2 = nes.reindex(columns=all_cols, fill_value=0)
    return es2, nes2, all_cols

def build_table(es_csv: str, nes_csv: str) -> pd.DataFrame:
    es = load_and_clean(es_csv)
    nes = load_and_clean(nes_csv)
    es, nes, cols = align_columns(es, nes)
    es_pct = normalize_percentage(es)
    nes_pct = normalize_percentage(nes)
    df = pd.DataFrame({"feature": cols, "ES_%": es_pct.values, "NES_%": nes_pct.values})
    # Enforce order: keep ORDER first, then append remaining in alphabetical order
    ordered = [f for f in ORDER if f in df["feature"].values]
    remaining = sorted([f for f in df["feature"].values if f not in ordered])
    final_order = ordered + remaining
    df = df.set_index("feature").loc[final_order].reset_index()
    return df

def plot_grouped(df: pd.DataFrame, out_png: str, title: str):
    x = np.arange(len(df))
    bw = 0.4
    plt.figure(figsize=(14, 6))
    plt.bar(x - bw/2, df["ES_%"], width=bw, label="ES")
    plt.bar(x + bw/2, df["NES_%"], width=bw, label="NES")
    plt.xticks(x, df["feature"], rotation=60, ha="right")
    plt.ylabel("Percentage (%)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    # Add value labels on bars
    for i, v in enumerate(df["ES_%"]):
        plt.text(i - bw/2, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(df["NES_%"]):
        plt.text(i + bw/2, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.savefig(out_png, dpi=200)
    plt.close()





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--es_csv", required=True, help="ES-only CSV")
    ap.add_argument("--nes_csv", required=True, help="NES-only CSV")
    ap.add_argument("--out_prefix", default="es_nes_percent", help="Output prefix for files")
    args = ap.parse_args()

    table = build_table(args.es_csv, args.nes_csv)
    out_csv = f"{args.out_prefix}.csv"
    out_png = f"{args.out_prefix}.png"
    table.to_csv(out_csv, index=False)
    plot_grouped(table, out_png, "Feature Percentages (ES vs NES) — Doctor")
    print(f"Wrote: {out_csv}\nWrote: {out_png}")

if __name__ == "__main__":
    main()
