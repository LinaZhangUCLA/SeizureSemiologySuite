import pandas as pd

def main():
    # 输入文件名，如不在当前目录下可改成完整路径
    input_csv = "/home/lina/ssb/SeizureSemiologyBench/metrics/task6_compare_metrics_with_human/task6_all_score.csv"

    # 读取数据
    df = pd.read_csv(input_csv)

    # 要和 human 列计算相关系数的指标列
    metrics = ["bleu_corpus", "rouge1_f1", "rougeL_f1", "berts_f1", "rqi"]
    target = "human"

    # 逐列计算皮尔逊相关系数并打印
    for m in metrics:
        if m not in df.columns:
            print(f"列 {m} 不在数据中，跳过")
            continue
        if target not in df.columns:
            raise ValueError(f"目标列 {target} 不在数据中")

        corr = df[m].corr(df[target])  # 默认是 Pearson 相关系数
        print(f"{m} 与 {target} 的皮尔逊相关系数: {corr:.4f}")

if __name__ == "__main__":
    main()
