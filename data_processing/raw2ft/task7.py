# ================================================================
# Created by: Jungang
# Email: ljungang.02@gmail.com
# Description: Prepare ms-swift fine-tuning JSON for task-7 groundtruth data.
# ================================================================
import csv
import json
from pathlib import Path
from datetime import datetime

DEFAULT_DATE = datetime.now().strftime("%Y-%m-%d")
output_path = Path(f"./dataset/ft_data/ft_task_7_{DEFAULT_DATE}.json")

# 文件路径
csv_esnes = Path("./fintune/data/task7_esnes_annotation.csv")
csv_gt = Path("./result/ground_truth/task7_annotation.csv")

def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def make_patient_id(filename: str):
    return filename.split("@")[0]

def make_video_id(filename: str):
    return filename.replace(".mp4", "")

# 加载CSV
records_a = load_csv(csv_esnes)  
records_b = load_csv(csv_gt)     # 只有 label

json_records = []

# ---------- (1) Type-A ----------
for row in records_a:
    print(row)
    file_name = row["\ufefffile_name"]
    label = row["label"].strip()
    report = row["report"].strip()

    entry = {
        "patient_id": make_patient_id(file_name),
        "video_id": make_video_id(file_name),
        "channel": "task-7-1",
        "messages": [
            {
                "role": "system",
                "content": "You are a medical assistant helping to observe, describe, and analyze seizure videos."
            },
            {
                "role": "user",
                "content": (
                    f"{report} "
                    "Based on the patient’s seizure video and seizure semiology report, "
                    "determine whether the patient has epileptic seizures (ES) or non-epileptic events (NES). "
                    "Answer with 'ES' or 'NES’ and do not include any other text."
                )
            },
            {
                "role": "assistant",
                "content": label
            }
        ],
        "videos": [f"./dataset/videos/task7_seizure_videos/{file_name}"],
    }
    json_records.append(entry)

# ---------- (2) Type-B ----------
for row in records_b:
    file_name = row["file_name"]
    label = row["label"].strip()
    # 找到对应报告
    report_match = next((r["report"] for r in records_a if r["\ufefffile_name"] == file_name), "")

    entry = {
        "patient_id": make_patient_id(file_name),
        "video_id": make_video_id(file_name),
        "channel": "task-7-2",
        "messages": [
            {
                "role": "system",
                "content": "You are a medical assistant helping to observe, describe, and analyze seizure videos."
            },
            {
                "role": "user",
                "content": (
                    "Describe the patient's seizure symptoms in the video and diagnose whether it is an epileptic seizure (ES) or a non-epileptic event (NES). "
                    "Provide a description and answer with 'ES' or 'NES’. "
                    "Respond with exactly one JSON object in the format { \"answer\": \"...\", \"description\": \"...\" } "
                    "and do not include any extra text outside of the JSON."
                )
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {"answer": label, "description": report_match},
                    ensure_ascii=False
                )
            }
        ],
        "videos": [f"./dataset/videos/task7_seizure_videos/{file_name}"],
    }
    json_records.append(entry)

# ---------- 保存 ----------
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(json_records, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(json_records)} entries to {output_path}")
