import pandas as pd
import os


feature_folders_split = [
    ['blank_stare', 'close_eyes', 'eye_blinking'],
    ['tonic', 'clonic', 'arm_flexion'],
    ['arm_straightening', 'figure4', 'oral_automatisms'],
    ['limb_automatisms', 'face_pulling'],
    ['head_turning', 'asynchronous_movement'],
    ['pelvic_thrusting', 'arms_move_simultaneously'],
    ['full_body_shaking', 'start'],
    ['face_twitching', 'end'],
    # 'ictal_vocalization', 'verbal_responsiveness','occur_during_sleep',
]
features = [f for group in feature_folders_split for f in group]

def parse_ts(ts):
    if pd.isna(ts):
        return None
    if isinstance(ts, (int, float)):
        return int(ts)
    s = str(ts).strip()
    if s.isdigit():
        return int(s)
    parts = s.split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None
    if len(parts) == 2:      # MM:SS
        m, sec = parts
        return m*60 + sec
    elif len(parts) == 3:    # H:MM:SS
        h, m, sec = parts
        return h*3600 + m*60 + sec
    return None


def calculate( input_csv):
    gt = pd.read_csv("/home/lina/ssb/SeizureSemiologyBench/result/ground_truth/Task4_time_annotation.csv")
    vlm = pd.read_csv(input_csv)    

    vlm["pred_time"] = vlm["timestamp"].apply(parse_ts)

    merged = pd.merge(gt, vlm, on=["file_name", "feature"], how="inner")
    merged = merged.dropna(subset=["pred_time"])
    merged["abs_err"] = (merged["target_time"] - merged["pred_time"]).abs()

    mae_dict = {}
    for feat in features:
        sub = merged[merged["feature"] == feat]
        if len(sub) == 0:
            mae_dict[feat] = 0
        else:
            mae = sub["abs_err"].mean()
            mae_dict[feat] = round(mae,2)
    print(mae_dict)
    return mae_dict





if __name__ == "__main__":
  
    # input_csv = "/home/lina/ssb/SeizureSemiologyBench/result/vlm_inference/Qwen2.5-VL-7B-Instruct/task5.csv"
    # out_file = process_csv(input_csv)
    # print(f"已生成: {out_file}") 
    # Model names
    base_dir = '/home/lina/ssb/SeizureSemiologyBench/result/vlm_inference/'
    model_names = [
        'Qwen2.5-VL-7B-Instruct',
        'InternVL3_5-8B',
        'Qwen2.5-VL-32B-Instruct',
        'InternVL3_5-38B',
        'Qwen2.5-VL-72B-Instruct',
        'audio-flamingo-3',
        'Qwen2.5-Omni-7B',
        #'Lingshu-32B',
        'Qwen3-VL-8B-Instruct',
        'Qwen3-VL-32B-Instruct',
    ]
    metric_rows = []
    for model in model_names:
        print(model)     
        input_csv = f"{base_dir}{model}/Task4_{model}_all.csv"
        if os.path.isfile(input_csv):
            mae_dict = calculate(input_csv)
            mae_dict["model"] = model
            metric_rows.append(mae_dict)
    
    out_df = pd.DataFrame(metric_rows, columns= (["model"] + features))
    out_path = f"/home/lina/ssb/SeizureSemiologyBench/metrics/task4_time_mae_metrics.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print("save to ", out_path)


    
  