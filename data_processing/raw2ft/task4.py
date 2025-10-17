# ================================================================
# Created by: Jungang
# Email: ljungang.02@gmail.com
# Description: Prepare ms-swift fine-tuning JSON for task-4 groundtruth data.
# ================================================================

import os
import csv
import json
import argparse
from datetime import datetime

# ---------- 路径配置 ----------
INPUT_DIR = "./fintune/task4_groundtruth"
OUTPUT_DIR = "./dataset/ft_data"
VIDEO_BASE_PATH = "./videos"
DEFAULT_DATE = datetime.now().strftime("%Y-%m-%d")
# ------------------------------

# ---------- feature definitions ----------
feature_definitions = {
   'arm_flexion': "flex their arms or arm at the elbows for at least a few video frames.",
   'arms_move_simultaneously': "move their arms approximately at the same time.",
   'occur_during_sleep': "are sleeping at the beginning of the video.",
   'close_eyes': "keep their eyes consistently or mostly closed.",
   'eye_blinking': "show rapid blinking of the eyes.",
   'tonic': "enter the tonic phase, showing sudden sustained stiffness or rigidity.",
   'clonic': "enter the clonic phase, showing rhythmic jerking movements.",
   'arm_straightening': "straighten or extend their arms at the elbow for a few frames.",
   'figure4': "form a 'Figure 4' posture (one arm flexed, one extended).",
   'oral_automatisms': "exhibit repetitive mouth or tongue movements like chewing or lip-smacking.",
   'limb_automatisms': "show repetitive limb movements such as rubbing or patting.",
   'face_pulling': "show unilateral sustained face-pulling movements.",
   'face_twitching': "show small twitches on the face muscles.",
   'head_turning': "forcibly rotate the head to one side.",
   'asynchronous_movement': "have limbs shaking asynchronously.",
   'pelvic_thrusting': "show rhythmic forward-backward hip movements.",
   'full_body_shaking': "experience shaking of the entire body.",
   'start': "start of a seizure event.",
   'end': "end of a seizure event."
}
# -----------------------------------------


def get_task4_feature_prompt(feature: str):
    """根据特征名称返回对应的任务提示（user prompt）"""
    if feature == "start":
        return '''
This video shows the start of a seizure event. Tell me the exact timestamp (MM:SS) when you first observe any seizure sign.
Return only the JSON format: {"timestamp": "MM:SS"}
'''.strip()
    elif feature == "end":
        return '''
This video shows the end of a seizure event. Tell me the exact timestamp (MM:SS) when you observe the last seizure sign.
Return only the JSON format: {"timestamp": "MM:SS"}
'''.strip()
    else:
        description = feature_definitions.get(feature, "")
        return f'''
This video shows when a patient {description}
Tell me the exact timestamp (MM:SS) when this symptom first appears.
Return only the JSON format: {{"timestamp": "MM:SS"}}
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
    feature = infer_feature_from_filename(csv_path)
    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row["file_name"].strip()
            target_time = row["target_time"].strip()
            if not file_name:
                continue

            patient_id = file_name.split("@")[0]
            video_id = os.path.splitext(file_name)[0]
            video_path = os.path.join(video_base_path, file_name)

            # 构造 prompt 与回答
            user_prompt = get_task4_feature_prompt(feature)
            mmss_time = seconds_to_mmss(target_time)
            assistant_answer = json.dumps({"timestamp": mmss_time}, ensure_ascii=False)

            sample = {
                "patient_id": patient_id,
                "video_id": video_id,
                "feature": feature,
                "target_time": target_time,
                "channel":"task-4",
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
    parser = argparse.ArgumentParser(description="Prepare ms-swift fine-tuning JSON for task4 groundtruth data.")
    parser.add_argument("--input_dir", default=INPUT_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--video_base_path", default=VIDEO_BASE_PATH)
    parser.add_argument("--date", default=DEFAULT_DATE)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_samples = []

    for fn in os.listdir(args.input_dir):
        if fn.endswith(".csv"):
            csv_path = os.path.join(args.input_dir, fn)
            samples = process_csv(csv_path, args.video_base_path)
            all_samples.extend(samples)

    out_path = os.path.join(args.output_dir, f"ft_task_4_{args.date}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(all_samples)} samples to {out_path}")


if __name__ == "__main__":
    main()
