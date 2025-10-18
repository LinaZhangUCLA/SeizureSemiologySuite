#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_patient_group_stratified_5fold.py

说明：
- 针对“438个视频，label含 ES / NES / Sample（仅6个）”的场景：
  * 自动忽略 label == 'Sample'（大小写不敏感）的样本；
  * 从 file_name 中提取病人ID（第一个'@'前的子串）；
  * 按“病人分组（防泄漏）+ 5折分层（保持 ES/NES 比例接近整体）”进行交叉验证划分；
  * 导出每折的 train/val CSV、整体汇总 CSV、被忽略样本清单，以及各折统计信息。

输入CSV至少包含：
- file_name：视频文件名，例如 A0002@5-13-2021@UA6693LK@sz_v1_1.mp4
- label：标签，可能是 ES / NES / Sample（大小写不敏感）

输出：
- out_dir/all_folds.csv：仅包含 ES/NES 的样本及其 fold（0~K-1 表示该样本所在的验证折）
- out_dir/fold{i}_train.csv、out_dir/fold{i}_val.csv：每折训练/验证子集（仅 ES/NES）
- out_dir/ignored_samples.csv：被忽略的 Sample 样本列表
- out_dir/fold_stats.csv：整体与各折的 ES/NES 数量与占比汇总

依赖：
- pandas, numpy
- scikit-learn（如环境支持 StratifiedGroupKFold 将优先使用；否则用脚本内置的回退算法）

用法：
python make_patient_group_stratified_5fold.py --csv data.csv --outdir out_dir \
  --filename-col file_name --label-col label --n-splits 5 --seed 42
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Tuple
import numpy as np
import pandas as pd


def extract_patient_id(file_name: str) -> str:
    """从文件名中提取病人ID：取第一个'@'之前的部分。"""
    if not isinstance(file_name, str):
        file_name = str(file_name)
    return file_name.split("@", 1)[0] if "@" in file_name else file_name


def filter_es_nes(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """将标签标准化为大写，并拆分出 ES/NES 与 Sample（忽略）。"""
    lab_upper = df[label_col].astype(str).str.strip().str.upper()
    is_sample = lab_upper == "SAMPLE"
    ignored = df[is_sample].copy()
    kept = df[~is_sample].copy()
    # 仅保留 ES/NES，其它情况直接报错（数据质量校验）
    kept_upper = kept[label_col].astype(str).str.strip().str.upper()
    bad = ~kept_upper.isin(["ES", "NES"])
    if bad.any():
        raise ValueError(f"检测到除 ES/NES/Sample 之外的标签：{sorted(kept[label_col][bad].unique())}")
    kept[label_col] = kept_upper
    ignored[label_col] = ignored[label_col].astype(str).str.strip()
    return kept, ignored


def try_stratified_group_kfold(n_splits: int, seed: int, y, groups):
    """若可用则返回 StratifiedGroupKFold.split 的生成器；否则返回 None。"""
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
    """
    自实现的“分组 + 分层”K折划分（贪心启发式）：
    - 目标：每个 group（病人）仅落入一个折；各折样本数与正类占比接近全局。
    - 正类定义：label == 'ES' (映射为1)，负类 'NES' (映射为0)。
    - 评分函数：
      score = |(es_after/total_after) - global_ratio| + λ * |(total_after - target_size)|/target_size
      其中 λ 默认 0.1。
    """
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
    """生成整体与各折的样本数与标签分布统计表（仅 ES/NES）。"""
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
    parser = argparse.ArgumentParser(description="按病人ID做5折分组+分层划分，忽略Sample，保持ES/NES比例接近整体。")
    parser.add_argument("--csv", required=True, help="输入CSV路径，需包含 file_name 与 label 列")
    parser.add_argument("--outdir", required=True, help="输出目录")
    parser.add_argument("--n-splits", type=int, default=5, help="折数（默认=5）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认=42）")
    parser.add_argument("--filename-col", default="file_name", help="文件名列名（默认=file_name）")
    parser.add_argument("--label-col", default="label", help="标签列名（默认=label）")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 读取
    df = pd.read_csv(args.csv)
    if args.filename_col not in df.columns:
        raise KeyError(f"未找到列：{args.filename_col}")
    if args.label_col not in df.columns:
        raise KeyError(f"未找到列：{args.label_col}")

    # 忽略 Sample，仅保留 ES/NES
    kept, ignored = filter_es_nes(df, args.label_col)

    # 保存被忽略的样本
    ignored_path = os.path.join(args.outdir, "ignored_samples.csv")
    ignored.to_csv(ignored_path, index=False, encoding="utf-8")
    print(f"[信息] 忽略的 Sample 条数：{len(ignored)}（已保存：{ignored_path}）")

    # 提取 patient_id
    kept = kept.copy()
    kept["patient_id"] = kept[args.filename_col].map(extract_patient_id)

    # 将标签映射为二值，ES=1, NES=0（用于分层）
    y = (kept[args.label_col].str.upper() == "ES").astype(int).values
    groups = kept["patient_id"].astype(str).values

    # 一些基本校验
    n_groups = kept["patient_id"].nunique()
    if n_groups < args.n_splits:
        raise ValueError(f"可用病人的数量（{n_groups}）少于折数（{args.n_splits}），无法完成 K 折分组。")

    # 优先使用 sklearn 的 StratifiedGroupKFold
    split_gen = try_stratified_group_kfold(args.n_splits, args.seed, y, groups)
    if split_gen is not None:
        fold_of_idx = np.full(len(kept), -1, dtype=int)
        for fold_id, (_, val_idx) in enumerate(split_gen):
            fold_of_idx[val_idx] = fold_id
        assert (fold_of_idx >= 0).all(), "存在未分配折号的样本"
    else:
        # 回退贪心
        fold_of_idx = greedy_group_stratified_split(y, groups, n_splits=args.n_splits, seed=args.seed)

    kept = kept.copy()
    kept["fold"] = fold_of_idx

    # 保存汇总
    all_path = os.path.join(args.outdir, "all_folds.csv")
    kept.to_csv(all_path, index=False, encoding="utf-8")
    print(f"[保存] {all_path}")

    # 导出每折 train/val
    for f in sorted(kept["fold"].unique()):
        f = int(f)
        val_df = kept[kept["fold"] == f].copy()
        train_df = kept[kept["fold"] != f].copy()
        val_path = os.path.join(args.outdir, f"fold{f}_val.csv")
        train_path = os.path.join(args.outdir, f"fold{f}_train.csv")
        val_df.to_csv(val_path, index=False, encoding="utf-8")
        train_df.to_csv(train_path, index=False, encoding="utf-8")
        print(f"[保存] fold {f}: {train_path} / {val_path}")

    # 统计
    stat = summarize_folds(kept, label_col=args.label_col)
    stat_path = os.path.join(args.outdir, "fold_stats.csv")
    stat.to_csv(stat_path, encoding="utf-8")
    print(f"[保存] {stat_path}")
    print("\n=== 统计预览（仅 ES/NES） ===")
    print(stat)


if __name__ == "__main__":
    main()
