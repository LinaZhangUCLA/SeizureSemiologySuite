# async_qwen_vllm_runner.py
import os, glob, json, asyncio, random
from typing import Dict, List
from openai import AsyncOpenAI

# ---------- config ----------
DATA_ROOT      = "/mnt/SSD3/junhua/testing_data"   # your prepared dataset root
MODEL          = "Qwen/Qwen2.5-VL-32B-Instruct"
MAX_IMAGES     = 60                        # cap to 60 per segment
MAX_TOKENS     = 256                      # modest for latency/VRAM
TEMPERATURE    = 0.0
CONCURRENCY    = 16                        # tune 8–16 for 4×A6000
TIMEOUT_S      = 180
RETRIES        = 2                         # light retry per request
OUT_FILENAME   = "answers.raw.jsonl"       # per-segment output file

BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY  = os.getenv("VLLM_API_KEY", "sk-local-secret")
client   = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

# ---------- prompts ----------
all_features = [
    "occur_during_sleep","blank_stare","close_eyes","eye_blinking",
    "tonic","clonic","arm_flexion","arm_straightening","figure4",
    "oral_automatisms","limb_automatisms","face_pulling","face_twitching",
    "head_turning","asynchronous_movement","pelvic_thrusting",
    "arms_move_simultaneously","full_body_shaking",
]

def get_prompts() -> Dict[str, str]:
    # paste your full mapping here (exact strings you provided)
    p = {}
    p["blank_stare"] = "Does the patient exhibit a blank stare? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON."
    p["arm_flexion"] = "Does the patient flex their arms or arm at the elbows for at least a few video frames? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON."
    p["arms_move_simultaneously"] = "Do the patient's arms start moving approximately at the same time? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON."
    p["occur_during_sleep"] = "Is the patient sleeping at the beginning of the video? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON."
    p["ictal_vocalization"] = "Does the patient make any groaning, moaning, guttural sounds or do they utter stereotyped repetitive phrases? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON."
    p["close_eyes"] = "Do the patient's eyes remain consistently closed or mostly closed throughout the video? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON."
    p["eye_blinking"] = "Does the patient show rapid blinking of the eyes during the video? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON."
    p["tonic"] = "The tonic phase is marked by a sudden onset of sustained stiffness or rigidity, usually lasting 5-20 seconds. This stiffness may be generalized, with all limbs held in fixed extension or flexion posture and can include stiffening of the head and axial body. It may also be focal involving a subset of body parts or just one body part at a time. Does this patient show tonic? Give an answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON."
    p["clonic"] = "Clonic Phase: Rhythmic, jerking muscle contractions involving the limbs, face, and trunk. The jerking movements gradually slow before ceasing entirely. Clonic movements present as rhythmic, regular stereotyped contraction and relaxation of the affected body parts. Does this patient show clonic? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON."
    p["arm_straightening"] = "Does the patient straighten or extend their arms or arm at the elbow for at least a few video frames? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON."
    p["figure4"] = "Figure 4 refers to a tonic sustained posture where one arm is flexed while the other is extended at the same time. Does the patient exhibit a Figure 4 posture? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON."
    p["oral_automatisms"] = "Does the patient exhibit repetitive, stereotyped mouth or tongue movements such as chewing, lip-smacking, or swallowing? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON."
    p["limb_automatisms"] = "Does the patient exhibit repetitive, stereotyped limb movements such as fumbling, picking, rubbing or patting?  Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON."
    p["face_pulling"] = "Does the patient exhibit unilateral sustained face-pulling movements? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON."
    p["face_twitching"] = "Are there small muscle twitches observed on the patient's face? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON."
    p["head_turning"] = "Does the patient forcibly or stiffly rotate their head to one side in the video? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON."
    p["asynchronous_movement"] = "Do you observe the patient's limbs shake with variable frequency or amplitude with respect to one another? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON."
    p["pelvic_thrusting"] = "Does the patient display repetitive, rhythmic, anteroposterior (forward-and-backward) movements of the hips? Answer 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON."
    p["full_body_shaking"] = "Does the patient experience shaking of the entire body including arms, legs, torso? Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format { \"answer\": \"...\", \"justification\": \"...\" } and do not include any extra text outside of the JSON."
    assert set(all_features).issubset(p.keys())
    return p

# ---------- message builder (images first = identical prefix across prompts) ----------
def build_message(frame_dir: str, prompt: str) -> List[dict]:
    frames = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))[:MAX_IMAGES]
    if not frames:
        raise FileNotFoundError(f"No frames in {frame_dir}")
    content = [{"type":"image_url","image_url":{"url":f"file://{p}"}} for p in frames]
    content.append({"type":"text","text":prompt})
    return [{"role":"user","content":content}]

async def ask_one(frame_dir: str, prompt: str) -> str:
    # tiny retry with backoff for robustness
    last_err = None
    delay = 1.0
    for _ in range(RETRIES + 1):
        try:
            messages = build_message(frame_dir, prompt)
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=MODEL, messages=messages,
                    temperature=TEMPERATURE, max_tokens=MAX_TOKENS),
                timeout=TIMEOUT_S
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            await asyncio.sleep(delay + random.random()*0.3)
            delay *= 2
    raise last_err

async def run_segment(segment_dir: str, prompts: Dict[str,str]):
    frames_dir = os.path.join(segment_dir, "frames")
    out_fp = os.path.join(segment_dir, OUT_FILENAME)

    # resume support: don’t repeat features already saved
    seen = set()
    if os.path.exists(out_fp):
        with open(out_fp, "r", encoding="utf-8") as f:
            for line in f:
                try: seen.add(json.loads(line)["feature"])
                except: pass

    sem = asyncio.Semaphore(CONCURRENCY)
    async def worker(feature: str, text: str):
        async with sem:
            raw = await ask_one(frames_dir, text)
            return {"feature": feature, "raw": raw}

    tasks = [worker(f, prompts[f]) for f in all_features if f not in seen]
    if not tasks:
        print(f"[skip] {segment_dir} done already")
        return

    # write as each finishes
    with open(out_fp, "a", encoding="utf-8") as out:
        for coro in asyncio.as_completed(tasks):
            rec = await coro
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
    print(f"[ok] {segment_dir} → {out_fp}")

def iter_segments(root: str):
    for vdir in sorted(glob.glob(os.path.join(root, "*"))):
        for seg in sorted(glob.glob(os.path.join(vdir, "segment_*"))):
            yield seg

async def main():
    prompts = get_prompts()
    for seg_dir in iter_segments(DATA_ROOT):
        print("processing:", seg_dir)
        try:
            await run_segment(seg_dir, prompts)
        except Exception as e:
            print("[warn]", seg_dir, "failed:", e)

if __name__ == "__main__":
    asyncio.run(main())
