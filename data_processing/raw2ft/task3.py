import os
import csv
import json
import argparse
from datetime import datetime

# ---------- 路径配置 ----------
INPUT_DIR = "../raw_data/task3_annotation.csv"
OUTPUT_DIR = "../ft_data"
VIDEO_BASE_PATH = "/mnt/SSD3/lina/finetune_videos"
# VIDEO_BASE_PATH = "./dataset/videos"
DEFAULT_DATE = datetime.now().strftime("%Y-%m-%d")
# ------------------------------




def get_task3_feature_prompt(feature: str):

    if feature == 'left_right_head_turning':
        return '''
        Does the patient's head turn to the patient's left or to the patient's right?
        Answer with \"left\" or \"right\" only. Do not include any extra text. Return exactly one word: left or right.
        '''.strip()
    if feature == 'left_right_arm_movement':
        return '''
        Which arm of the patient is moving in the video?
        Answer with \"left\" or \"right\" only. Do not include any extra text. Return exactly one word: left or right.
        '''.strip()
    if feature == 'body_region_onset':
        return '''
        Localize which body part shows the earliest visible seizure sign.
        Answer only with one of the following options: head, eyes, mouth, face, left arm, left leg, right arm, right leg, arms,legs,full body.
        '''.strip()



def seconds_to_mmss(seconds_str: str):
    """把秒数字符串转换为MM:SS格式"""
    try:
        s = float(seconds_str)
        minutes = int(s // 60)
        seconds = int(round(s % 60))
        return f"{minutes:02d}:{seconds:02d}"
    except Exception:
        return "N/A"


def infer_feature_from_filename(csv_path: str):
    base = os.path.basename(csv_path)
    name = os.path.splitext(base)[0]
    for suffix in ["_timestamps", "_timestamp"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def process_csv(csv_path, video_base_path):
    """处理单个CSV，返回样本列表"""
    samples = []
    all_features = ['left_right_head_turning', 'left_right_arm_movement', 'body_region_onset']
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row["file_name"].strip()

            if not file_name:
                continue



            for feature in all_features:

                patient_id = file_name.split("@")[0]
                video_id = os.path.splitext(file_name)[0]
                if feature == "left_right_arm_movement":
                    video_path = os.path.join(video_base_path, "task3_arm_movement")
                elif feature == "left_right_head_turning":
                    video_path = os.path.join(video_base_path, "task3_head_turning")
                elif feature == "body_region_onset":
                    video_path = os.path.join(video_base_path, "body_region_onset")
                video_path = os.path.join(video_path, file_name)


                raw_answer = row[feature].strip()
                if raw_answer is not None and raw_answer != "N/A":

                    # 构造 prompt 与回答
                    user_prompt = get_task3_feature_prompt(feature)
                    assistant_answer = raw_answer


                    sample = {
                        "patient_id": patient_id,
                        "video_id": video_id,
                        "feature": feature,
                        "channel":"task-3",
                        "messages": [
                            {"role": "system", "content": "You are a medical assistant helping to observe, describe, and analyze seizure videos."},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": assistant_answer},
                        ],
                        "videos": [video_path],
                    }
                    samples.append(sample)
    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare ms-swift fine-tuning JSON for task3 data.")
    parser.add_argument("--input_dir", default=INPUT_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--video_base_path", default=VIDEO_BASE_PATH)
    parser.add_argument("--date", default=DEFAULT_DATE)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_samples = []

    samples = process_csv(args.input_dir, args.video_base_path)
    all_samples.extend(samples)

    out_path = os.path.join(args.output_dir, f"ft_task_3_{args.date}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(all_samples)} samples to {out_path}")


if __name__ == "__main__":
    main()
