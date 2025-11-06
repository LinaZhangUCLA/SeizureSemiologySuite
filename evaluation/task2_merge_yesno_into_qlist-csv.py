#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
from collections import OrderedDict

def _rename_first_col_to_filename(df: pd.DataFrame) -> pd.DataFrame:
    """把第一列统一命名为 file_name（兼容 'file name'）"""
    first = df.columns[0]
    if first != "file_name":
        df = df.rename(columns={first: "file_name"})
    # safe for typo'file name'
    if "file name" in df.columns and "file_name" not in df.columns:
        df = df.rename(columns={"file name": "file_name"})
    return df

def _group_features_by_qids(qlist_cols):
    """
    从 question-list 的列名中，按 feature 分组（feature = 去掉末尾 _Q\d+ 的前缀）
    返回 OrderedDict[feature -> [qid1, qid2, ...]]，顺序与 question-list 中首次出现顺序一致
    """
    groups = OrderedDict()
    for c in qlist_cols:
        if c == "file_name":
            continue
        if "_Q" in c:
            feat = c.rsplit("_Q", 1)[0]
            groups.setdefault(feat, []).append(c)
    return groups

def main():
    ap = argparse.ArgumentParser(
        "Merge yes/no feature columns from inputs CSV into question-list CSV, placing each yes/no column before its QIDs."
    )
    ap.add_argument("--inputs", required=True, help="原始 inputs CSV（含 yes/no 与 justification 列）; original input raw csv")
    ap.add_argument("--qlist",  required=True, help="question-list CSV（由你的问答脚本输出的）; the question-list csv")
    ap.add_argument("--out-dir", required=True, help="输出目录;output dir")
    ap.add_argument("--out-name", default="merged.csv", help="输出文件名（默认 merged.csv）")
    ap.add_argument("--fill-missing", default="", help="当 inputs 缺失某 feature 列时的填充值（默认空串，可改为 'unknown'）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out_name)

    df_in = pd.read_csv(args.inputs, dtype=str)
    df_q  = pd.read_csv(args.qlist,  dtype=str)
    df_in = _rename_first_col_to_filename(df_in)
    df_q  = _rename_first_col_to_filename(df_q)

    # taking the intersection
    common = pd.Index(df_in["file_name"]).intersection(pd.Index(df_q["file_name"]))
    if common.empty:
        raise ValueError("两个 CSV 在 file_name 上没有交集，无法合并。")

    # using as question-list oder
    df_q = df_q[df_q["file_name"].isin(common)].copy()
    order = {fn: i for i, fn in enumerate(df_q["file_name"].tolist())}
    df_in = df_in[df_in["file_name"].isin(common)].copy()
    df_in["__ord__"] = df_in["file_name"].map(order)
    df_in.sort_values("__ord__", inplace=True)
    df_in.drop(columns="__ord__", inplace=True)
    df_in.reset_index(drop=True, inplace=True)
    df_q.reset_index(drop=True, inplace=True)

    groups = _group_features_by_qids(df_q.columns)

    out = pd.DataFrame({"file_name": df_q["file_name"]})

    in_idx = df_in.set_index("file_name")

    for feat, qids in groups.items():
        if feat in in_idx.columns:
            out[feat] = in_idx[feat].reindex(out["file_name"]).values
        else:
            out[feat] = args.fill_missing

        for qid in qids:
            if qid in df_q.columns:
                out[qid] = df_q[qid]
            else:
                out[qid] = ""

    # 
    out.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    print(f"Rows: {len(out)}  Cols: {len(out.columns)}")
    print("Columns order preview:", ", ".join(list(out.columns)[:10]), " ...")

if __name__ == "__main__":
    main()
