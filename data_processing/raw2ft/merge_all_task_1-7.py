# ================================================================
# Created by: Jungang
# Email: ljungang.02@gmail.com
# Description: Merge all JSON files under ./data_processing/ft_data into ./dataset/sft_merge_xxx.json.
# ================================================================


import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from datetime import datetime
DEFAULT_DATE = datetime.now().strftime("%Y-%m-%d")

SRC_DIR = Path("./data_processing/ft_data")
OUT_PATH = Path(f"./dataset/sft_merge_{DEFAULT_DATE}.json")
DEDUPE = False  # change to True if you want to deduplicate

def load_json_file(p: Path) -> Iterable[Dict[str, Any]]:
    """Load a JSON file that may contain a list, a single object, or JSONL."""
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Try whole-file JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            return [obj]
        # Unexpected but tolerable: wrap as empty
        return []
    except json.JSONDecodeError:
        # Fallback to JSON Lines (one JSON object per line)
        items: List[Dict[str, Any]] = []
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict):
                    items.append(rec)
                elif isinstance(rec, list):
                    items.extend([x for x in rec if isinstance(x, dict)])
            except json.JSONDecodeError:
                print(f"[WARN] Skip invalid JSON in {p} line {i}")
        return items

def dedupe_key(obj: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    return (
        obj.get("patient_id"),
        obj.get("video_id"),
        obj.get("feature"),
        obj.get("channel"),
    )

def main() -> None:
    if not SRC_DIR.exists():
        raise SystemExit(f"[ERROR] Source directory not found: {SRC_DIR.resolve()}")

    files = sorted(SRC_DIR.rglob("*.json"))
    if not files:
        raise SystemExit(f"[ERROR] No JSON files found under: {SRC_DIR.resolve()}")

    merged: List[Dict[str, Any]] = []
    for fp in files:
        try:
            records = list(load_json_file(fp))
            merged.extend(records)
            print(f"[OK] {fp} -> +{len(records)}")
        except Exception as e:
            print(f"[ERROR] {fp}: {e}")

    if DEDUPE:
        seen = set()
        unique: List[Dict[str, Any]] = []
        for rec in merged:
            k = dedupe_key(rec)
            if k not in seen:
                seen.add(k)
                unique.append(rec)
        print(f"[INFO] Deduped {len(merged)} -> {len(unique)}")
        merged = unique

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Wrote {len(merged)} records to {OUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
