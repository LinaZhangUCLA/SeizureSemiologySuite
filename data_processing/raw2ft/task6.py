import csv
import json
import os
from pathlib import Path
from datetime import datetime
DEFAULT_DATE = datetime.now().strftime("%Y-%m-%d")
def get_task6_prompt():
    return """
    Write a concise semiology description for this **seizure video**,
    focusing ONLY on observable **patient signs**.
    HARD RESTRICTIONS
    - Do NOT mention staff, restraints, bed/blanket/pillow, room devices, cameras, EEG leads/overlays, or timestamps.
    - Avoid vague words like “agitation”, “restlessness”, “discomfort”, or “adjusting position”.
    WHAT TO COVER (include an item ONLY if it is clearly visible in this video)
    • Early signs: blank stare, lip smacking, right/left head version, unilateral arm flexion/extension, tonic stiffening, clonic jerks.
    • Evolution & laterality: e.g., fencer posturing (left flexion with right extension), spread to bilateral tonic–clonic, automatisms, asynchronous shaking.
    STYLE
    - Write **1–3 short sentences in English only**, specific and minimal.
    - Examples (no labels): “Blank stare, then rightward head version with right arm extension; later bilateral tonic–clonic.”
        “Left arm flexion with right extension (fencer); rhythmic jerks follow. Unresponsive at the end.”
    Output ONLY the paragraph (no lists, no headers, no JSON).
    """

input_csv = "./result/ground_truth/task6_report_annotation.csv"
output_path = Path(f"./dataset/ft_data/ft_task_6_{DEFAULT_DATE}.json")

records = []
with open(input_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row)
        video_filename = row["\ufefffile_name"]
        description = row["report"]

        patient_id = video_filename.split("@")[0]
        video_id = os.path.splitext(video_filename)[0]

        record = {
            "patient_id": patient_id,
            "video_id": video_id,
            "channel": "task-6",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a medical assistant helping to observe, describe, and analyze seizure videos."
                },
                {
                    "role": "user",
                    "content": get_task6_prompt().strip()
                },
                {
                    "role": "assistant",
                    "content": description.strip()
                }
            ],
            "videos": [f"./videos/{video_filename}"]
        }

        records.append(record)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print(f"✅ JSON saved to {output_path}, total {len(records)} entries.")
