# ================================================================
# Created by: Jungang
# Email: ljungang.02@gmail.com
# Description: processor for qwen2.5-omni fine-tuning on seizure semiology bench tasks.
# ================================================================
import os, json, pathlib
from typing import Any, Dict, List, Optional

# ---------------- ms-swift 导入 ----------------
from swift.llm.dataset.preprocessor.core import ResponsePreprocessor
from swift.llm.dataset.register import DatasetMeta, register_dataset

# TASK_DATASET = os.environ.get("TASK_DATASET", "./dataset/ft_data/ft_task_5_2025-10-23.jsonl")

TASK_DATASET = os.environ.get("TASK_DATASET", './dataset/sft_merge_2025-10-30_swift_train.jsonl')

# ===================== 工具函数 =====================
def _abspath(p: str) -> str:
    return os.path.abspath(p)

def _probe_video_meta(vpath: str) -> Dict[str, float]:
    """用 decord 只读取元信息（总帧数/原fps），不抽帧；失败则给默认值。"""
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(vpath, ctx=cpu(0))
        total = float(len(vr))
        try:
            raw_fps = float(vr.get_avg_fps())
            if raw_fps <= 0:
                raw_fps = 30.0
        except Exception:
            raw_fps = 30.0
        return {"total": total, "raw_fps": raw_fps}
    except Exception:
        return {"total": 0.0, "raw_fps": 30.0}

def _rule_by_channel(channel: str) -> Dict[str, Any]:
    ch = (channel or "")
    if "task-4" in ch:
        return {"mode": "sample_fps", "sample_fps": 1.0}
    if "task-7-1" in ch or "task-7-2" in ch or "task-7" in ch:
        return {"mode": "nframes", "nframes": 60}
    return {"mode": "nframes", "nframes": 60}
    return {"mode": "sample_fps", "sample_fps": 2.0}

def _make_videos_kwargs(vpath: str, channel: str) -> Dict[str, Any]:
    rule = _rule_by_channel(channel)
    if rule["mode"] == "sample_fps":
        return {"sample_fps": float(rule["sample_fps"])}
    # 均匀采样目标 nframes，同时估算 sample_fps（便于后端按 fps 采样时接近 nframes）
    meta = _probe_video_meta(vpath)
    total, raw_fps = max(meta["total"], 1.0), meta["raw_fps"]
    nframes = int(rule.get("nframes", 60))
    est_sample_fps = max((nframes / total) * raw_fps, 0.1)
    return {"nframes": nframes, "sample_fps": float(est_sample_fps)}

# ===================== 数据预处理 =====================
class ChannelAwarePreprocessor(ResponsePreprocessor):
    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        msgs = row.get("messages")
        if not isinstance(msgs, list) or len(msgs) == 0:
            return None
        norm = []
        for m in msgs:
            if not isinstance(m, dict): 
                continue
            role = m.get("role")
            content = m.get("content")
            if isinstance(content, list):
                picked = None
                for it in content:
                    if isinstance(it, dict) and it.get("type") == "text" and isinstance(it.get("text"), str):
                        picked = it["text"]; break
                content = picked if picked is not None else json.dumps(content, ensure_ascii=False)
            elif isinstance(content, dict):
                content = json.dumps(content, ensure_ascii=False)
            elif content is None:
                content = ""
            else:
                content = str(content)
            content = content.strip()
            if role and content:
                norm.append({"role": role, "content": content})
        if not norm:
            return None

        vids = row.get("videos") or []
        if isinstance(vids, str):
            vids = [vids]
        if not vids:
            return None
        vpath = _abspath(vids[0])
        channel = row.get("channel", "")

        vkw = _make_videos_kwargs(vpath, channel)
        return {
            "messages": norm,
            "videos":   [vpath],
            "videos_kwargs": [vkw],
            "channel":  channel,
        }

# 注册数据集
register_dataset(DatasetMeta(
    dataset_path=TASK_DATASET,
    dataset_name='ssb',
    preprocess_func=ChannelAwarePreprocessor(),
))

