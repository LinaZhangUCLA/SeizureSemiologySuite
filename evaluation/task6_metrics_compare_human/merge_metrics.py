import pandas as pd

def merge_rqi_to_metrics(
    metrics_csv="/home/lina/ssb/SeizureSemiologyBench/metrics/compare_with_human_score.csv",
    rqi_csv="/home/lina/ssb/SeizureSemiologyBench/metrics/task6_20videos_rqi_comparison.csv",
    out_csv="/home/lina/ssb/SeizureSemiologyBench/metrics/compare_with_human_score_with_rqi.csv"
):
    # 读取第一个 csv（包含 bleu、rouge、berts 等）
    df_metrics = pd.read_csv(metrics_csv)

    # 读取第二个 csv（包含 rqi）
    df_rqi = pd.read_csv(rqi_csv)

    # 只保留对齐需要的列（假设第二个文件里列名为 model, file_name, rqi）
    #df_rqi = df_rqi[["model", "file_name", "rqi"]]
    df_rqi = df_rqi.melt(
        id_vars=["file_name"],      # 不动的列
        var_name="model",           # 原来的列名（各模型）变成这一列
        value_name="human",           # rqi 数值
    )

    df_rqi["human"] = df_rqi["human"].astype(float).round(1)
    # 按 model 和 file_name 对齐，把 rqi 合并到第一个表中
    df_merged = df_metrics.merge(
        df_rqi,
        on=["model", "file_name"],
        how="left",   # 如果有没对上的，会是 NaN
        validate="one_to_one"  # 确保是一一对应，否则会报错提醒你
    )

    # 可选：检查一下总行数是不是 60
    if len(df_merged) != 60:
        print(f"警告：合并后的行数是 {len(df_merged)}，不是预期的 60 行，请检查匹配情况。")

    # 保存结果
    df_merged.to_csv(out_csv, index=False)
    print(f"合并完成，已保存到：{out_csv}")


if __name__ == "__main__":
    merge_rqi_to_metrics(metrics_csv="/home/lina/ssb/SeizureSemiologyBench/metrics/compare_with_human_score_with_rqi.csv",
    rqi_csv="/home/lina/ssb/SeizureSemiologyBench/metrics/task6_human_score.csv",
    out_csv="/home/lina/ssb/SeizureSemiologyBench/metrics/task6_all_score.csv")
