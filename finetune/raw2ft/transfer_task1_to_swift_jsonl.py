# ================================================================
# Created by: Jungang
# Email: ljungang.02@gmail.com
# Description: Convert merged json file into Swift-friendly format json/jsonl,
#              preserving original fields and injecting <video>/<image> tags.
# ================================================================


import json
from pathlib import Path
from typing import Any, Dict, Iterable, List
from datetime import datetime


#!/usr/bin/env python3
import sys
import pandas as pd


test_path = "./datasplit/fold0_val.csv"
df = pd.read_csv(test_path)  
uniq_patitent_list = sorted(df["patient_id"].dropna().astype(str).unique())
print(len(uniq_patitent_list))


DEFAULT_DATE = datetime.now().strftime("%Y-%m-%d")
#DEFAULT_DATE_Pre = '2025-10-28'pyt

IN_PATH  = Path(f"./ft_data/ft_task_1only_2026-03-17.json")
OUT_JSON = Path(f"./dataset/sft_task1only_{DEFAULT_DATE}_swift_train.json")
OUT_JSONL= Path(f"./dataset/sft_task1only_{DEFAULT_DATE}_swift_train.jsonl")

def load_mixed_json(p: Path) -> Iterable[Dict[str, Any]]:
    """Load a file that may be a JSON array, single JSON object, or JSONL."""
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
        return []
    except json.JSONDecodeError:
        items: List[Dict[str, Any]] = []
        for ln, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
                if isinstance(rec, dict):
                    items.append(rec)
                elif isinstance(rec, list):
                    items.extend([x for x in rec if isinstance(x, dict)])
            except json.JSONDecodeError:
                print(f"[WARN] skip invalid JSON at line {ln}")
        return items

def normalize_messages(obj: Dict[str, Any]) -> List[Dict[str, str]]:
    """Pick and validate the conversation messages."""
    msgs = obj.get("messages", [])
    if not isinstance(msgs, list):
        return []
    norm: List[Dict[str, str]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if isinstance(role, str) and isinstance(content, str):
            norm.append({"role": role, "content": content})
    return norm

def normalize_media_list(val: Any) -> List[str]:
    """Ensure media field is a list of strings."""
    if val is None:
        return []
    if isinstance(val, str):
        return [val] if val.strip() else []
    if isinstance(val, list):
        out = []
        for x in val:
            if isinstance(x, str) and x.strip():
                out.append(x)
        return out
    return []

def inject_media_tags(messages: List[Dict[str, str]],
                      num_videos: int,
                      num_images: int) -> List[Dict[str, str]]:
    """
    Ensure the last user message contains enough <video>/<image> tokens
    to match counts of attached videos/images. If tokens already present,
    only top up the difference.
    """
    if not messages:
        return messages

    # find the last user message
    user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user" and isinstance(messages[i].get("content"), str):
            user_idx = i
            break

    if user_idx is None:
        return messages

    content = messages[user_idx]["content"]
    if not isinstance(content, str):
        return messages

    # Count existing tokens
    existing_video = content.count("<video>")
    existing_image = content.count("<image>")

    # Prepare top-up tokens
    need_video = max(0, num_videos - existing_video)
    need_image = max(0, num_images - existing_image)

    if need_video or need_image:
        suffix_tokens: List[str] = []
        if need_image:
            suffix_tokens.extend(["<image>"] * need_image)
        if need_video:
            suffix_tokens.extend(["<video>"] * need_video)
        # Append with a preceding newline for readability
        joiner = " "
        content = content.rstrip() + ("\n\n" if not content.endswith("\n") else "") + joiner.join(suffix_tokens)
        messages[user_idx]["content"] = content

    return messages

def main() -> None:
    if not IN_PATH.exists():
        raise SystemExit(f"[ERROR] Input not found: {IN_PATH.resolve()}")

    records = list(load_mixed_json(IN_PATH))
    print(f"[INFO] loaded {len(records)} raw records")

    converted: List[Dict[str, Any]] = []
    dropped = 0
    not_exit = 0
    for r in records:
        #print("patientid: ",r.get("patient_id"))
        if r.get("patient_id") in uniq_patitent_list:
            dropped += 1
            continue
        msgs = normalize_messages(r)
        if not msgs:
            dropped += 1
            continue

        videos = normalize_media_list(r.get("videos"))
        images = normalize_media_list(r.get("images"))

        msgs = inject_media_tags(msgs, num_videos=len(videos), num_images=len(images))

        # Preserve original fields as much as possible
        out_item: Dict[str, Any] = dict(r)
        out_item["messages"] = msgs
        out_item["videos"] = videos
        out_item["images"] = images
        #print(out_item)
        #print("videos: ", videos)
        p = Path(videos[0])
        if p.is_file():  # 存在且是文件
            converted.append(out_item)   
        else:    
            print("not exist: ",videos)
            #print(videos[0])
            not_exit += 1
            continue
        
        
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    #JSON array
    OUT_JSON.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
    # JSONL
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[DONE] wrote {len(converted)} records to:")
    print(f"  - {OUT_JSON.resolve()}")
    print(f"  - {OUT_JSONL.resolve()}")
    if dropped:
        print(f"[NOTE] dropped {dropped} records without valid messages")
    if not_exit:
        print(f"[NOTE] {not_exit} records not exist")     

if __name__ == "__main__":
    main()
