# ================================================================
# Created by: Jungang
# Email: ljungang.02@gmail.com
# Description: Prepare ms-swift fine-tuning JSON for task-5 groundtruth data.
# ================================================================
import csv
import json
import os
from pathlib import Path
from datetime import datetime
VIDEO_BASE_PATH = "./dataset/videos/task56_segments"
DEFAULT_DATE = datetime.now().strftime("%Y-%m-%d")
# 定义 prompt
def get_task5_prompt():
    return '''
    Output the sequence of the any observed seizure symptoms of the patient in the video in chronological order.
    The symptoms are limited to head_turning, blank_stare, close_eyes, eye_blinking, face_pulling, face_twitching, tonic, clonic, arm_straightening, arm_flexion, figure4, oral_automatisms, limb_automatisms, asynchronous_movement, pelvic_thrusting, full_body_shaking, arms_move_simultaneously.
    If a symptom is not present in the video, it should not be included in the output.
    Example output: head_turning, arm_straightening, arm_flexion, tonic, clonic.
    Output only the seizure symptoms. Do not include any other text.
    '''

# 输入与输出路径
csv_path = Path("../rawdata/task5_segment_sequence_annotation.csv")
output_path = Path(f"../ft_data/ft_task_5_{DEFAULT_DATE}.json")

# 定义症状关键词，用于自动提取 ground truth
SYMPTOMS = [
    "head_turning", "blank_stare", "close_eyes", "eye_blinking", "face_pulling", 
    "face_twitching", "tonic", "clonic", "arm_straightening", "arm_flexion", 
    "figure4", "oral_automatisms", "limb_automatisms", "asynchronous_movement", 
    "pelvic_thrusting", "full_body_shaking", "arms_move_simultaneously"
]

def extract_symptoms(description: str):
    """
    简单的症状提取规则：在描述中匹配关键词（大小写不敏感）
    """
    description = description.lower()
    found = [s for s in SYMPTOMS if s.replace("_", " ") in description or s in description]
    return ", ".join(found)

# 读取CSV并生成样本
samples = []
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if len(row) < 1:
            continue
        
        # video_id,  description = row
        # print(description)
        #symptoms = extract_symptoms(row["segment_feature_list"])
        file_name = row["segment_video_name"].strip()
        video_path = os.path.join(VIDEO_BASE_PATH, file_name)
    
        patient_id = file_name.split("@")[0]
        sample = {
                "patient_id": patient_id,
                "video_id": os.path.splitext(file_name)[0],
                "channel":"task-5",
                "messages": [
                    {"role": "system", "content": "You are a medical assistant helping to observe, describe, and analyze seizure videos."},
                    {"role": "user", "content": get_task5_prompt().strip()},
                    # {"role": "assistant", "content": symptoms if symptoms else "none"},
                    {"role": "assistant", "content": str(row["segment_feature_list"]).replace('""', '"').replace('"', '') if row["segment_feature_list"] else ""},
                ],
                "videos": [video_path],
                
            }
        samples.append(sample)

# 写出JSONL文件
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print(f"✅ 已生成 {len(samples)} 条样本，保存至 {output_path}")
