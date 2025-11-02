#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NES Doctor vs. NES Novices Percentage Plotter
---------------------------------------------
输入:
  --doc_csv   NES doctors 数据 (0/1 特征标记)
  --nov_csv   NES novices 数据 (0/1 特征标记)
  --order_txt 特征顺序文件 (一行一个特征名，可选)
  --out_prefix 输出文件前缀 (默认 nes_doc_vs_nov)

输出:
  1. {out_prefix}.csv   各特征百分比表
  2. {out_prefix}.png   对比条形图 (带数值标注)
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_clean(path):
    drop_cols = ["Unnamed: 0", "file_name", "es_flag", "nes_flag"]
    df = pd.read_csv(path)
    df = df[[c for c in df.columns if c not in drop_cols]].copy()
    rename_map = {"verbar": "verbal_responsiveness", "arm_flex": "arm_flexion"}
    df = df.rename(columns=rename_map)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc_csv", required=True, help="ES doctors CSV")
    ap.add_argument("--nov_csv", required=True, help="ES novices CSV")
    ap.add_argument("--order_txt", default=None, help="特征顺序文件")
    ap.add_argument("--out_prefix", default="nes_doc_vs_nov", help="输出前缀")
    args = ap.parse_args()

    # 读入并清理
    doc = load_and_clean(args.doc_csv)
    nov = load_and_clean(args.nov_csv)

    # 对齐列
    all_feats = sorted(set(doc.columns).union(set(nov.columns)))
    doc = doc.reindex(columns=all_feats, fill_value=0)
    nov = nov.reindex(columns=all_feats, fill_value=0)

    # 各自总数
    n_doc, n_nov = len(doc), len(nov)

    # 计算百分比
    doc_pct = doc.sum() / max(n_doc, 1) * 100.0
    nov_pct = nov.sum() / max(n_nov, 1) * 100.0

    pct_df = pd.DataFrame({
        "feature": all_feats,
        "ES_doctors_%": doc_pct.values,
        "ES_novices_%": nov_pct.values
    })

    # 加载顺序
    if args.order_txt and os.path.exists(args.order_txt):
        with open(args.order_txt, "r", encoding="utf-8") as f:
            order = [line.strip() for line in f if line.strip()]
    else:
        order = []
    ordered = [f for f in order if f in pct_df["feature"].values]
    remaining = sorted([f for f in pct_df["feature"].values if f not in ordered])
    final_order = ordered + remaining
    pct_df = pct_df.set_index("feature").loc[final_order].reset_index()

    # 导出 CSV
    out_csv = f"{args.out_prefix}.csv"
    pct_df.to_csv(out_csv, index=False)

    # 画图
    x = np.arange(len(pct_df))
    bw = 0.4
    plt.figure(figsize=(max(12, 0.5 * len(pct_df) + 4), 7))

    bars1 = plt.bar(x - bw/2, pct_df["ES_doctors_%"], width=bw, label="ES doctors")
    bars2 = plt.bar(x + bw/2, pct_df["ES_novices_%"], width=bw, label="ES novices")

    # 数值标注
    for b in bars1:
        v = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    for b in bars2:
        v = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.xticks(x, pct_df["feature"], rotation=60, ha="right")
    plt.ylabel("Percentage (%)")
    plt.title("ES Doctor vs. ES Novices")
    plt.legend()
    plt.tight_layout()

    out_png = f"{args.out_prefix}.png"
    plt.savefig(out_png, dpi=220)
    plt.close()

    print(f"✅ 已生成: {out_csv}, {out_png}")

if __name__ == "__main__":
    main()
