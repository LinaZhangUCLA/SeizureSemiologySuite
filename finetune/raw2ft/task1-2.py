import os
import csv
import json
import argparse
from datetime import datetime

# ---------- 路径配置 ----------
INPUT_DIR = "../rawdata/task12_finetune_featrue_videos.csv"
OUTPUT_DIR = "../ft_data"
#VIDEO_BASE_PATH = "/mnt/SSD3/lina/finetune_videos"
VIDEO_BASE_PATH = "./dataset/videos"
DEFAULT_DATE = datetime.now().strftime("%Y-%m-%d")
# ------------------------------




def get_task1_feature_prompt(feature: str):

    if feature == 'blank_stare':
        return '''
        Does the patient exhibit a blank stare? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON.
        '''.strip()
    if feature == 'arm_flexion':
        return '''
        Does the patient flex their arms or arm at the elbows for at least a few video frames? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON.
        '''.strip()
    if feature == 'arms_move_simultaneously':
        return '''
        Do the patient's arms start moving approximately at the same time? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON.
        '''.strip()
    if feature == 'occur_during_sleep':
        return '''
        Is the patient sleeping at the beginning of the video? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON.
        '''.strip()
    if feature == 'ictal_vocalization':
        return '''
        Does the patient make any groaning, moaning, guttural sounds or do they utter stereotyped repetitive phrases? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON.
        '''.strip()
    if feature == 'close_eyes':
        return '''
        Do the patient's eyes remain consistently closed or mostly closed throughout the video? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON.
        '''.strip()
    if feature == 'eye_blinking':
        return '''
        Does the patient show rapid blinking of the eyes during the video? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON.
        '''.strip()
    if feature == 'tonic':
        return '''
        The tonic phase is marked by a sudden onset of sustained stiffness or rigidity, usually lasting 5-20 seconds. This stiffness may be generalized, with all limbs held in fixed extension or flexion posture and can include stiffening of the head and axial body. It may also be focal involving a subset of body parts or just one body part at a time. Does this patient show tonic? Give an answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON.
        '''.strip()
    if feature == 'clonic':
        return '''
        Clonic Phase: Rhythmic, jerking muscle contractions involving the limbs, face, and trunk. The jerking movements gradually slow before ceasing entirely. Clonic movements present as rhythmic, regular stereotyped contraction and relaxation of the affected body parts. Does this patient show clonic? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON.
        '''.strip()
    if feature == 'arm_straightening':
        return '''
        Does the patient straighten or extend their arms or arm at the elbow for at least a few video frames? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON.
        '''.strip()
    if feature == 'figure4':
        return '''
        Figure 4 refers to a tonic sustained posture where one arm is flexed while the other is extended at the same time. Does the patient exhibit a Figure 4 posture? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON.
        '''.strip()
    if feature == 'oral_automatisms':
        return '''
        Does the patient exhibit repetitive, stereotyped mouth or tongue movements such as chewing, lip-smacking, or swallowing? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON.
        '''.strip()
    if feature == 'limb_automatisms':
        return '''
        Does the patient exhibit repetitive, stereotyped limb movements such as fumbling, picking, rubbing or patting?  Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON.
        '''.strip()
    if feature == 'face_pulling':
        return '''
        Does the patient exhibit unilateral sustained face-pulling movements? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON.
        '''.strip()
    if feature == 'face_twitching':
        return '''
        Are there small muscle twitches observed on the patient's face? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON.
        '''.strip()
    if feature == 'head_turning':
        return '''
        Does the patient forcibly or stiffly rotate their head to one side in the video? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON.
        '''.strip()
    if feature == 'asynchronous_movement':
        return '''
        Do you observe the patient's limbs shake with variable frequency or amplitude with respect to one another? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON.
        '''.strip()
    if feature == 'pelvic_thrusting':
        return '''
        Does the patient display repetitive, rhythmic, anteroposterior (forward-and-backward) movements of the hips? Answer 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON.
        '''.strip()
    if feature == 'full_body_shaking':
        return '''
        Does the patient experience shaking of the entire body including arms, legs, torso? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON.
        '''.strip()
    if feature == 'verbal_responsiveness':    
        return '''
        If the patient is addressed verbally by a different person, did they respond verbally in a coherent manner? Answer 'yes' or 'no'. If the patient is not addressed verbally by a different person, then the answer should be 'NA'.Give an answer and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON.
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
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row["file_name"].split("/")[-1].strip()
            feature = row["feature"].strip()
            if not file_name:
                continue

            patient_id = file_name.split("@")[0]
            video_id = os.path.splitext(file_name)[0]
            video_path = os.path.join(video_base_path, "task12_features")
            video_path = os.path.join(video_path, feature)
            video_path = os.path.join(video_path, file_name)



            # 构造 prompt 与回答
            user_prompt = get_task1_feature_prompt(feature)
            assistant_answer1 = row["answer"].strip()
            assistant_answer2 = row["justification"].strip()

            # { \"answer\": \"...\", \"justification\": \"...\" }

            sample = {
                "patient_id": patient_id,
                "video_id": video_id,
                "feature": feature,
                "channel":"task-1-2",
                "messages": [
                    {"role": "system", "content": "You are a medical assistant helping to observe, describe, and analyze seizure videos."},
                    {"role": "user", "content": user_prompt},
                    #{"role": "assistant", "content": str({ "answer": assistant_answer1, "justification": assistant_answer2 })},
                    {"role": "assistant", "content": json.dumps({"answer": assistant_answer1, "justification": assistant_answer2},ensure_ascii=False)},
                ],
                "videos": [video_path],
            }
            samples.append(sample)
    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare ms-swift fine-tuning JSON for task1 data.")
    parser.add_argument("--input_dir", default=INPUT_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--video_base_path", default=VIDEO_BASE_PATH)
    parser.add_argument("--date", default=DEFAULT_DATE)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_samples = []

    samples = process_csv(args.input_dir, args.video_base_path)
    all_samples.extend(samples)
    print("total sample num: ", len(all_samples))
    out_path = os.path.join(args.output_dir, f"ft_task_1-2_{args.date}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(all_samples)} samples to {out_path}")


if __name__ == "__main__":
    main()
