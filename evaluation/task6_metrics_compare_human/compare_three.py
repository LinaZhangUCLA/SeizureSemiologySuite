import pandas as pd

import itertools
import numpy as np




def main():
    # 1. 读入数据
    df = pd.read_csv("/home/lina/ssb/SeizureSemiologyBench/metrics/task6_compare_metrics_with_human/task6_all_score.csv")

    scores = df["human"].values
    idx_pairs = list(itertools.combinations(range(len(df)), 2))

    def pairwise_acc(metric):
        m = df[metric].values
        agree = 0
        total = 0
        for i, j in idx_pairs:
            dh = scores[i] - scores[j]
            dm = m[i] - m[j]
            if dh == 0:
                continue  # 人类认为一样高，跳过或者另算
            if np.sign(dh) == np.sign(dm):
                agree += 1
            total += 1
        return agree / total if total > 0 else np.nan

    # 2. 要计算的指标列
    metrics = ["bleu_corpus", "rouge1_f1", "rougeL_f1", "berts_f1", "rqi"]
    target = "human"

    # 3. 逐列计算 Pearson / Spearman / Kendall，并存到字典里
    result = {}

    for m in metrics:
        if m not in df.columns:
            print(f"列 {m} 不在数据中，跳过")
            continue
        if target not in df.columns:
            raise ValueError(f"目标列 {target} 不在数据中")

        pearson = df[m].corr(df[target], method="pearson")
        spearman = df[m].corr(df[target], method="spearman")
        kendall = df[m].corr(df[target], method="kendall")
        pairwise_accuracy = pairwise_acc(m)

        result[m] = {
            "pearson": round(pearson,2),
            "spearman": round(spearman,2),
            "kendall": round(kendall,2),
            "pairwise_accuracy": round(pairwise_accuracy,2),
        }

    # 4. 结果转成 DataFrame：行是 metric，列是三种系数
    corr_df = pd.DataFrame.from_dict(result, orient="index")
    corr_df.index.name = "metric"

    # 打印看一眼
    print(corr_df)

    # 5. 保存成 csv
    corr_df.to_csv("/home/lina/ssb/SeizureSemiologyBench/metrics/task6_compare_metrics_with_human/task6_all_score_corr_4.csv")

if __name__ == "__main__":
    main()
