#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_group_stratified_folds.py

用途：
- 读取包含列 file_name 和 label 的CSV文件（label=ES/NES），从 file_name 中提取病人ID（第一个'@'之前），
- 按“病人ID分组 + 5折交叉验证”进行划分，防止病人泄漏，
- 同时尽量保持每一折的 ES/NES 比例接近整体数据集，
- 输出每一折的 train/val CSV 以及包含所有样本及其 fold 的汇总CSV，并保存统计信息。

特点：
- 优先使用 scikit-learn 的 StratifiedGroupKFold（若可用）。
- 如果当前环境没有 StratifiedGroupKFold，则使用自带的贪心分配算法作为回退实现。

使用示例：
python make_group_stratified_folds.py --csv data.csv --outdir out_dir \
    --n-splits 5 --seed 42 --filename-col file_name --label-col label

输入CSV至少包含：
- file_name：视频文件名，例如 A0002@5-13-2021@UA6693LK@sz_v1_1.mp4
- label：类别，建议为 ES / NES（大小写不敏感）

输出：
- out_dir/all_folds.csv：包含每条样本的 fold（0~K-1 为验证折）
- out_dir/fold{i}_train.csv、out_dir/fold{i}_val.csv：第 i 折的训练/验证划分
- out_dir/fold_stats.csv：整体与每折的样本数、各类别计数与占比

作者：你未来的自己
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Tuple
import numpy as np
import pandas as pd

# -----------------------------
# 工具函数
# -----------------------------
def extract_patient_id(file_name: str) -> str:
    """从文件名中提取病人ID：取第一个'@'之前的部分。"""
    if not isinstance(file_name, str):
        file_name = str(file_name)
    return file_name.split("@", 1)[0] if "@" in file_name else file_name


def normalize_label(series: pd.Series, positive_label: str = "ES") -> Tuple[pd.Series, Dict[str, int], str]:
    """
    将标签标准化为二值 {0,1}，并返回映射与最终被视为正类的名称。
    - 默认把大小写不敏感的 'ES' 当作正类 1，'NES' 当作负类 0。
    - 如果数据不是 ES/NES（二分类），则使用 factorize 生成 0/1 标签，并以出现频次较少的类作为“正类”。
    """
    s = series.astype(str).str.strip()
    s_upper = s.str.upper()

    unique_upper = set(s_upper.unique())
    if {"ES", "NES"}.issuperset(unique_upper) and len(unique_upper) <= 2:
        y = (s_upper == positive_label.upper()).astype(int)
        mapping = {positive_label.upper(): 1, ("NES" if positive_label.upper()=="ES" else "ES"): 0}
        pos_name = positive_label.upper()
        return y, mapping, pos_name

    # 通用二分类回退逻辑：
    cats, uniques = pd.factorize(s)  # 0..n-1
    if len(uniques) != 2:
        raise ValueError(f"当前脚本仅支持二分类。检测到的类别为：{list(uniques)}")
    # 统计两个类别的频次，较少者设为正类（通常更关注少数类）
    counts = pd.Series(cats).value_counts().sort_index().tolist()  # [count_class0, count_class1]
    if counts[0] <= counts[1]:
        pos_idx = 0
    else:
        pos_idx = 1
    y = (cats == pos_idx).astype(int)
    mapping = {str(uniques[pos_idx]): 1, str(uniques[1 - pos_idx]): 0}
    pos_name = str(uniques[pos_idx])
    return y, mapping, pos_name


def try_stratified_group_kfold(n_splits: int, seed: int, y, groups):
    """
    若可用则返回 StratifiedGroupKFold.split 的生成器；否则返回 None。
    """
    try:
        from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        # X 参数不参与划分逻辑，可传索引
        idx = np.arange(len(y))
        return sgkf.split(idx, y, groups)
    except Exception:
        return None


def greedy_group_stratified_split(
    y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int
):
    """
    自实现的“分组 + 分层”5折划分（贪心启发式）：
    - 目标：在保证每个 group（病人）只落入一个折的前提下，使各折样本数与正类占比接近整体。
    - 思路：
        1) 先按“组内类别不平衡程度”从大到小排序（困难组先分配）。
        2) 依次把每个组放到当前“最合适”的折里。衡量准则是：
           score = |(es_after/total_after) - global_ratio| + λ * |(total_after - target_size)|/target_size
           其中 λ 默认 0.1。
    返回：一个长度为 N 的数组 fold_idx，每个样本的折号 [0..K-1]（表示验证折）。
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    assert n == len(groups)

    # 汇总每个组的 (样本数, 正类数)
    df = pd.DataFrame({"group": groups, "y": y})
    gstat = df.groupby("group")["y"].agg(["count", "sum"]).rename(columns={"count": "n", "sum": "n_pos"})
    # “不平衡度”= |2*n_pos - n|，越大越难分配
    gstat["imbalance"] = (2 * gstat["n_pos"] - gstat["n"]).abs()
    # 随机扰动以打破平手（保证可复现）
    gstat["rand"] = rng.rand(len(gstat))
    gstat = gstat.sort_values(by=["imbalance", "n", "rand"], ascending=[False, False, True])

    total_n = n
    total_pos = int(y.sum())
    global_ratio = total_pos / total_n if total_n > 0 else 0.0
    target_size = total_n / n_splits
    lam = 0.1  # 尺度平衡权重

    # 每折的当前累积
    fold_n = np.zeros(n_splits, dtype=int)
    fold_pos = np.zeros(n_splits, dtype=int)

    group_to_fold: Dict[str, int] = {}

    for g, row in gstat.iterrows():
        g_n = int(row["n"])
        g_pos = int(row["n_pos"])
        best_fold = None
        best_score = None

        # 在所有折中选择最优
        for f in range(n_splits):
            n_after = fold_n[f] + g_n
            pos_after = fold_pos[f] + g_pos
            if n_after == 0:
                ratio_after = 0.0
            else:
                ratio_after = pos_after / n_after
            ratio_term = abs(ratio_after - global_ratio)
            size_term = abs(n_after - target_size) / max(1.0, target_size)
            score = ratio_term + lam * size_term

            if (best_score is None) or (score < best_score):
                best_score = score
                best_fold = f

        # 绑定到该折
        assert best_fold is not None
        group_to_fold[g] = int(best_fold)
        fold_n[best_fold] += g_n
        fold_pos[best_fold] += g_pos

    # 生成样本级的折号
    fold_idx = np.array([group_to_fold[g] for g in groups], dtype=int)
    return fold_idx


def make_splits(
    df: pd.DataFrame,
    filename_col: str,
    label_col: str,
    n_splits: int = 5,
    seed: int = 42,
    positive_label: str = "ES",
) -> pd.DataFrame:
    """主流程：新增 patient_id 和 fold 列，返回带有折号的 DataFrame。"""
    if filename_col not in df.columns:
        raise KeyError(f"未找到列：{filename_col}")
    if label_col not in df.columns:
        raise KeyError(f"未找到列：{label_col}")

    df = df.copy()
    # 提取病人ID
    df["patient_id"] = df[filename_col].map(extract_patient_id)

    # 规范化标签为二值 y（正类=ES，或其他二分类）
    y, mapping, pos_name = normalize_label(df[label_col], positive_label=positive_label)
    df["_ybin_"] = y

    # 组信息
    groups = df["patient_id"].astype(str).values
    y_np = df["_ybin_"].astype(int).values

    # 优先尝试 StratifiedGroupKFold
    split_gen = try_stratified_group_kfold(n_splits=n_splits, seed=seed, y=y_np, groups=groups)

    if split_gen is not None:
        # 使用官方实现
        fold_of_idx = np.full(len(df), -1, dtype=int)
        for fold_id, (_, val_idx) in enumerate(split_gen):
            fold_of_idx[val_idx] = fold_id
        assert (fold_of_idx >= 0).all(), "存在未分配折号的样本"
    else:
        # 回退到贪心分配
        fold_of_idx = greedy_group_stratified_split(y_np, groups, n_splits=n_splits, seed=seed)

    df["fold"] = fold_of_idx
    return df.drop(columns=["_ybin_"])


def summarize_folds(df: pd.DataFrame, label_col: str, pos_label_name: str = "ES") -> pd.DataFrame:
    """生成整体与各折的样本数与标签分布统计表。"""
    def label_counts(frame: pd.DataFrame) -> pd.Series:
        cnt = frame[label_col].astype(str).str.upper().value_counts()
        # 统一暴露 ES / NES 两列（若不存在则填 0）
        es = int(cnt.get("ES", 0))
        nes = int(cnt.get("NES", 0))
        total = len(frame)
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

    stat = pd.DataFrame(rows)
    return stat


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="按病人ID做5折分组+分层划分，防止病人泄漏并保持ES/NES比例。")
    parser.add_argument("--csv", required=True, help="输入CSV路径，需包含 file_name 与 label 列")
    parser.add_argument("--outdir", required=True, help="输出目录")
    parser.add_argument("--n-splits", type=int, default=5, help="折数（默认=5）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认=42）")
    parser.add_argument("--filename-col", default="file_name", help="文件名列名（默认=file_name）")
    parser.add_argument("--label-col", default="label", help="标签列名（默认=label）")
    parser.add_argument("--positive-label", default="ES", help="作为正类的标签名（默认=ES；仅用于二值转换与统计显示）")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 读取数据
    df = pd.read_csv(args.csv)
    # 生成折号
    df_folds = make_splits(
        df,
        filename_col=args.filename_col,
        label_col=args.label_col,
        n_splits=args.n_splits,
        seed=args.seed,
        positive_label=args.positive_label,
    )

    # 保存所有样本及其折号
    all_path = os.path.join(args.outdir, "all_folds.csv")
    df_folds.to_csv(all_path, index=False, encoding="utf-8")
    print(f"[保存] {all_path}")

    # 逐折导出 train/val
    for f in sorted(df_folds["fold"].unique()):
        f = int(f)
        val_df = df_folds[df_folds["fold"] == f].copy()
        train_df = df_folds[df_folds["fold"] != f].copy()
        val_path = os.path.join(args.outdir, f"fold{f}_val.csv")
        train_path = os.path.join(args.outdir, f"fold{f}_train.csv")
        val_df.to_csv(val_path, index=False, encoding="utf-8")
        train_df.to_csv(train_path, index=False, encoding="utf-8")
        print(f"[保存] fold {f}: {train_path} / {val_path}")

    # 统计并保存
    stat = summarize_folds(df_folds, label_col=args.label_col, pos_label_name=args.positive_label)
    stat_path = os.path.join(args.outdir, "fold_stats.csv")
    stat.to_csv(stat_path, encoding="utf-8")
    print(f"[保存] {stat_path}")
    print("\n=== 统计预览 ===")
    print(stat)

if __name__ == "__main__":
    main()
