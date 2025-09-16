import os
import re
import csv
import json
import hashlib
import argparse
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

# decord for video decoding
from decord import VideoReader, cpu

# LMDeploy (PyTorch backend) + VLM utils
from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.utils import encode_image_base64

LOG = False

# --------------------------- Defaults ---------------------------
DEFAULT_MODEL_CACHE_DIR = '/mnt/SSD3/xinyi/benchmark/model_cache'
DEFAULT_OUTPUT_DIR = '/mnt/SSD3/xinyi/benchmark/output'

FPS = 2.0
MAX_FRAMES = 60
MAX_NEW_TOKENS = 2048

# --------------------------- CLI ---------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description='Seizure Video Feature Extraction using InternVL3.5 (LMDeploy, decord + base64)')

    parser.add_argument('--gpu', type=str, default='0', help='CUDA visible devices, e.g., 0 or 0,1')
    parser.add_argument('--model_name', type=str, default='OpenGVLab/InternVL3_5-8B', help='HF repo id')
    parser.add_argument('--tp', type=int, default=1, help='Tensor parallel degree for PyTorch backend')

    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Root dir containing: task1_segments, task4_head_turning, task4_arm_movement, task5_segment')

    parser.add_argument('--cache_dir', type=str, default=DEFAULT_MODEL_CACHE_DIR, help='Model cache root')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Output dir for CSVs')

    parser.add_argument('--videos_range', type=str, default='1-2314', help='1-indexed inclusive range, e.g., 1-100')

    return parser.parse_args()

# --------------------------- Prompts ---------------------------

def get_task3_prompt():
    return (
        """
        Output the sequence of the any observed seizure symptoms of the patient in the video in chronological order.
        The symptoms are limited to head_turning, blank_stare, close_eyes, eye_blinking, face_pulling, face_twitching,
        tonic, clonic, arm_straightening, arm_flexion, figure4, oral_automatisms, limb_automatisms,
        asynchronous_movement, pelvic_thrusting, full_body_shaking, arms_move_simultaneously.
        If a symptom is not present in the video, it should not be included in the output.
        Example output: head_turning, arm_straightening, arm_flexion, tonic, clonic.
        Output only the seizure symptoms. Do not include any other text.
        """
    )


def get_task6_prompt():
    return (
        """
        Generate a detailed report for this seizure video, describing the patient's observable actions, signs, and overall condition.
        The report must focus exclusively on the patient; do not include any descriptions of medical staff.
        For example: The patient is sleeping in bed. He lets out a loud groan and has versive head turn to the right. He has right upper extremity extension with left upper extremity flexion, followed by tonic-clonic activity. Later, the patient is unable to remember events preceding this seizure.
        Output the report as a cohesive paragraph in plain language. Do not include other content.
        """
    )


def get_task4_HT_prompt():
    return (
        """
        Does the patient's head turn to the patient's left or to the patient's right?
        Answer with "left" or "right" only. Do not include any extra text. Return exactly one word: left or right.
        """
    )


def get_task4_AM_prompt():
    return (
        """
        Which arm of the patient is moving in the video?
        Answer with "left" or "right" only. Do not include any extra text. Return exactly one word: left or right.
        """
    )


def get_task4_L_prompt():
    return (
        """
        Localize which body part shows the earliest visible seizure sign.
        Answer only with one of the following options: head, eyes, mouth, face, left arm, left leg, right arm, right leg, arms, legs, full body.
        """
    )


def get_task5_prompt():
    return (
        """
        This is a seizure clip. Provide the start time and end time of the seizure event if it is present in this video clip.
        If the seizure has already started at the beginning of the video, use "N/A" for the start time.
        If the seizure has not ended by the end of the video, use "N/A" for the end time.
        Please return the result in the following JSON format: { start_time: MM:SS or N/A, end_time: MM:SS or N/A }.
        Do not include any other text.
        """
    )

# --------------------------- Cache & Paths ---------------------------

def ensure_dirs(cache_root: str, output_dir: str):
    hf_cache_dir = os.path.join(cache_root, 'huggingface')
    lmdeploy_cache_dir = os.path.join(cache_root, 'lmdeploy')
    video_cache_dir = os.path.join(cache_root, 'video_cache')
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.makedirs(lmdeploy_cache_dir, exist_ok=True)
    os.makedirs(video_cache_dir, exist_ok=True)

    os.environ['HF_HOME'] = hf_cache_dir
    os.environ['TRANSFORMERS_CACHE'] = hf_cache_dir
    os.environ['HF_HUB_CACHE'] = hf_cache_dir
    os.environ['LMDEPLOY_CACHE_DIR'] = lmdeploy_cache_dir

    os.makedirs(output_dir, exist_ok=True)
    return hf_cache_dir, lmdeploy_cache_dir, video_cache_dir

# --------------------------- Video decoding (decord) ---------------------------

def _indices_by_fixed_fps(vr: VideoReader, target_fps: float, max_frames: int) -> List[int]:
    n_total = len(vr)
    if n_total <= 0:
        return []
    try:
        native_fps = float(vr.get_avg_fps())
        if not np.isfinite(native_fps) or native_fps <= 0:
            native_fps = target_fps
    except Exception:
        native_fps = target_fps

    frames_per_sample = native_fps / target_fps
    if frames_per_sample < 1.0:
        frames_per_sample = 1.0

    indices = []
    pos = 0.0
    last = -1
    while len(indices) < max_frames:
        idx = int(round(pos))
        if idx >= n_total:
            break
        if idx > last:
            indices.append(idx)
            last = idx
        pos += frames_per_sample

    if indices and indices[-1] >= n_total:
        indices = [i for i in indices if i < n_total]
    return indices


def get_video_frames_pil(video_file_path: str, target_fps: float = FPS, max_frames: int = MAX_FRAMES) -> List[Image.Image]:
    if video_file_path.startswith('file://'):
        video_file_path = video_file_path[7:]
    vr = VideoReader(video_file_path, ctx=cpu(0), num_threads=1)
    idxs = _indices_by_fixed_fps(vr, target_fps=target_fps, max_frames=max_frames)
    frames = [Image.fromarray(vr[i].asnumpy()).convert('RGB') for i in idxs]
    return frames

# --------------------------- LMDeploy Inference (IMAGE_TOKEN + base64) ---------------------------

def build_pipe(model_name: str, tp: int):
    engine_cfg = PytorchEngineConfig(tp=tp, session_len=32768)
    return pipeline(model_name, backend_config=engine_cfg)


def infer_with_messages(pipe, pil_frames: List[Image.Image], prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    # Compose text with IMAGE_TOKEN placeholders
    question = ''.join([f"Frame{i+1}: {IMAGE_TOKEN}\n" for i in range(len(pil_frames))]) + prompt

    # Content: text first, then images as base64 data URLs
    content = [{'type': 'text', 'text': question}]
    for img in pil_frames:
        content.append({
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{encode_image_base64(img)}',
                'max_dynamic_patch': 1
            }
        })

    messages = [dict(role='user', content=content)]
    gen_cfg = GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False)
    out = pipe(messages, gen_config=gen_cfg)

    if hasattr(out, 'text'):
        return out.text
    if isinstance(out, (list, tuple)) and len(out) > 0:
        return str(out[0])
    return str(out)


def inference(pipe, video_path: str, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    pil_frames = get_video_frames_pil(video_path, target_fps=FPS, max_frames=MAX_FRAMES)
    if len(pil_frames) == 0:
        return ""
    return infer_with_messages(pipe, pil_frames, prompt, max_new_tokens=max_new_tokens)

# --------------------------- Normalizers & Parsers ---------------------------

def normalize_direction_task4(ans: str) -> str:
    if ans is None:
        return 'fail'
    s = str(ans).strip().lower()
    if s in {'left', 'right'}:
        return s
    hits = re.findall(r'\b(left|right)\b', s)
    if hits:
        return hits[-1]
    if re.fullmatch(r'[lr]', s):
        return 'left' if s == 'l' else 'right'
    return 'fail'


def normalize_direction_task4_L(ans: str) -> str:
    if ans is None:
        return 'fail'
    s = str(ans).strip().lower()
    allowed = {
        'head', 'eyes', 'mouth', 'face', 'left arm', 'left leg', 'right arm', 'right leg', 'arms', 'legs', 'full body'
    }
    if s in allowed:
        return s
    hits = re.findall(r'\b(head|eyes|mouth|face|left arm|left leg|right arm|right leg|arms|legs|full body)\b', s)
    if hits:
        return hits[-1]
    return 'fail'


def parse_json_task5(vlm_output: str):
    try:
        m = re.search(r'\{.*\}', vlm_output, re.DOTALL)
        parsed = json.loads(m.group(0)) if m else json.loads(vlm_output)
        if isinstance(parsed, dict) and 'start_time' in parsed and 'end_time' in parsed:
            return parsed
        return {'start_time': 'N/A', 'end_time': 'N/A'}
    except Exception:
        return {'start_time': 'N/A', 'end_time': 'N/A'}


def format_time_task5(raw_output: str) -> str:
    raw = (raw_output or '').strip()
    if raw.upper() == 'N/A':
        return 'N/A'
    timestamp = None
    m = re.search(r'\[(.*?)\]', raw)
    if m:
        timestamp = m.group(1)
    if not timestamp:
        m = re.search(r'(\d{1,2}:\d{2}(?:\.\d+)?)', raw)
        if m:
            timestamp = m.group(1)
    if not timestamp:
        m = re.search(r'at\s+(\d{1,2}:\d{2}(?:\.\d+)?)', raw, re.IGNORECASE)
        if m:
            timestamp = m.group(1)
    if not timestamp:
        return 'N/A'
    try:
        mm, ss = timestamp.split(':')
        minutes = int(mm)
        seconds = int(float(ss))
        total = max(0, int(round(minutes * 60 + seconds)))
        m2, s2 = divmod(total, 60)
        return f"{m2:02}:{s2:02}"
    except Exception:
        return 'N/A'

# --------------------------- Task wrappers ---------------------------

def query_task3_6(pipe, video_fp: str, log_file_fp: str = None):
    raw_seq = inference(pipe, video_fp, get_task3_prompt())
    clip_seq_text = '"' + (raw_seq or '').strip() + '"'

    raw_rep = inference(pipe, video_fp, get_task6_prompt())
    clip_report_text = '"' + (raw_rep or '').strip() + '"'

    if LOG and log_file_fp:
        if not os.path.exists(log_file_fp):
            with open(log_file_fp, 'w', newline='', encoding='utf-8') as f:
                f.write('video_fp,event_sequence,report\n')
        with open(log_file_fp, 'a', newline='', encoding='utf-8') as f:
            f.write(f"{video_fp},{clip_seq_text},{clip_report_text}\n")
    return clip_seq_text, clip_report_text


def query_task4(pipe, video_fp: str, prompt: str) -> str:
    raw_answer = inference(pipe, video_fp, prompt)
    return normalize_direction_task4(raw_answer)


def query_task4_L(pipe, video_fp: str, prompt: str) -> str:
    raw_answer = inference(pipe, video_fp, prompt)
    return normalize_direction_task4_L(raw_answer)


def query_task5(pipe, video_fp: str):
    time_resp = inference(pipe, video_fp, get_task5_prompt())
    parsed = parse_json_task5(time_resp or '')
    start_time = format_time_task5(parsed.get('start_time', 'N/A'))
    end_time = format_time_task5(parsed.get('end_time', 'N/A'))
    return start_time, end_time

# --------------------------- File helpers ---------------------------

def get_fp_list(file_dir: str) -> List[str]:
    fps = []
    if not os.path.isdir(file_dir):
        return fps
    for name in sorted(set(os.listdir(file_dir))):
        if name.lower().endswith('.mp4'):
            fps.append(os.path.join(file_dir, name))
    return fps


def validate_videos_range(clip_files: List[str], videos_range: List[str]) -> List[int]:
    rng = [int(videos_range[0]), int(videos_range[1])]
    if len(rng) != 2:
        raise ValueError('videos_range must have exactly two numbers (start-end)')
    if rng[0] > rng[1]:
        raise ValueError('videos_range[0] must be <= videos_range[1]')
    if rng[0] < 1:
        print('Warning: start index < 1; setting to 1')
        rng[0] = 1
    if rng[1] > len(clip_files):
        print(f'Warning: end index > number of videos ({len(clip_files)}); setting to {len(clip_files)}')
        rng[1] = len(clip_files)
    return rng

# --------------------------- Main ---------------------------

def main():
    args = parse_arguments()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_name = args.model_name
    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        raise ValueError('--dataset_dir is required')

    # Subfolders per task
    task3_6_dataset_dir = os.path.join(dataset_dir, 'task1_segments')
    task4_HT_dataset_dir = os.path.join(dataset_dir, 'task4_head_turning')
    task4_AM_dataset_dir = os.path.join(dataset_dir, 'task4_arm_movement')
    task5_dataset_dir = os.path.join(dataset_dir, 'task5_segment')

    videos_range = args.videos_range.split('-')
    task3_6_videos_range = args.videos_range.split('-')
    task4_HT_videos_range = '1-130'.split('-')
    task4_AM_videos_range = '1-113'.split('-')
    # task4_HT_videos_range = '1-2'.split('-')
    # task4_AM_videos_range = '1-2'.split('-')
    task5_videos_range = args.videos_range.split('-')

    # Caches & output
    _, _, _ = ensure_dirs(os.environ.get('HF_HOME', DEFAULT_MODEL_CACHE_DIR), args.output_dir)

    # CSVs
    model_tag = model_name.split('/')[-1]
    task3_6_result_csv_fp = os.path.join(args.output_dir, f'Task3_6_{model_tag}_{task3_6_videos_range[0]}-{task3_6_videos_range[1]}.csv')
    task4_HT_result_csv_fp = os.path.join(args.output_dir, f'Task4_HT_{model_tag}_{task4_HT_videos_range[0]}-{task4_HT_videos_range[1]}.csv')
    task4_AM_result_csv_fp = os.path.join(args.output_dir, f'Task4_AM_{model_tag}_{task4_AM_videos_range[0]}-{task4_AM_videos_range[1]}.csv')
    task4L_5_result_csv_fp = os.path.join(args.output_dir, f'Task4L_5_{model_tag}_{task5_videos_range[0]}-{task5_videos_range[1]}.csv')

    # Build pipeline
    print('Initializing LMDeploy pipeline...')
    #pipe = build_pipe(model_name, tp=args.tp)
    pipe = None

    # Gather files
    task3_6_clip_fps = get_fp_list(task3_6_dataset_dir)
    task4_HT_clip_fps = get_fp_list(task4_HT_dataset_dir)
    task4_AM_clip_fps = get_fp_list(task4_AM_dataset_dir)
    task5_clip_fps = get_fp_list(task5_dataset_dir)
    print(len(task3_6_clip_fps), len(task4_HT_clip_fps), len(task4_AM_clip_fps), len(task5_clip_fps))

    # Task 3 + 6
    t36_rng = validate_videos_range(task3_6_clip_fps, task3_6_videos_range)
    for video_clip_fp in tqdm(task3_6_clip_fps[t36_rng[0]-1: t36_rng[1]], desc='Processing Task 3 and Task 6'):
        video_name = os.path.basename(video_clip_fp)
        if not os.path.exists(task3_6_result_csv_fp):
            with open(task3_6_result_csv_fp, 'w', newline='', encoding='utf-8') as f:
                f.write('video_name,event_sequence,report\n')
        with open(task3_6_result_csv_fp, 'r', encoding='utf-8') as f:
            if video_name in f.read():
                print(f'Skip (exists): {video_name}')
                continue
        log_file_fp = None
        if LOG:
            log_dir = os.path.join(args.output_dir, 'log')
            os.makedirs(log_dir, exist_ok=True)
            log_file_fp = os.path.join(log_dir, f'{video_name}.csv')
        try:
            raise NotImplementedError("Task 3 & 6 inference is disabled temporarily.")
            event_seq, report_text = query_task3_6(pipe, video_clip_fp, log_file_fp)
            with open(task3_6_result_csv_fp, 'a', newline='', encoding='utf-8') as f:
                f.write(f'{video_name},{event_seq},{report_text}\n')
        except Exception as e:
            print(f'Error (Task3+6) {video_name}: {e}')
            with open(task3_6_result_csv_fp, 'a', newline='', encoding='utf-8') as f:
                f.write(f'{video_name},"",""\n')
    print(f'Task 3+6 complete -> {task3_6_result_csv_fp}')

    # Task 4 (only if end > 2300)
    try:
        end_idx = int(videos_range[1])
    except Exception:
        end_idx = 0

    if end_idx > 2300:
        # HT
        ht_rng = validate_videos_range(task4_HT_clip_fps, task4_HT_videos_range)
        for video_clip_fp in tqdm(task4_HT_clip_fps[ht_rng[0]-1: ht_rng[1]], desc='Processing Task 4 HT'):
            video_name = os.path.basename(video_clip_fp)
            if not os.path.exists(task4_HT_result_csv_fp):
                with open(task4_HT_result_csv_fp, 'w', newline='', encoding='utf-8') as f:
                    f.write('video_name,head_turning_direction\n')
            with open(task4_HT_result_csv_fp, 'r', encoding='utf-8') as f:
                if video_name in f.read():
                    print(f'Skip (exists): {video_name}')
                    continue
            try:
                raise NotImplementedError("Task 4 HT inference is disabled temporarily.")
                ans = query_task4(pipe, video_clip_fp, get_task4_HT_prompt())
                with open(task4_HT_result_csv_fp, 'a', newline='', encoding='utf-8') as f:
                    f.write(f'{video_name},{ans}\n')
            except Exception as e:
                print(f'Error (Task4 HT) {video_name}: {e}')
                with open(task4_HT_result_csv_fp, 'a', newline='', encoding='utf-8') as f:
                    f.write(f'{video_name},N/A\n')

        # AM
        am_rng = validate_videos_range(task4_AM_clip_fps, task4_AM_videos_range)
        for video_clip_fp in tqdm(task4_AM_clip_fps[am_rng[0]-1: am_rng[1]], desc='Processing Task 4 AM'):
            video_name = os.path.basename(video_clip_fp)
            if not os.path.exists(task4_AM_result_csv_fp):
                with open(task4_AM_result_csv_fp, 'w', newline='', encoding='utf-8') as f:
                    f.write('video_name,arm_movement_direction\n')
            with open(task4_AM_result_csv_fp, 'r', encoding='utf-8') as f:
                if video_name in f.read():
                    print(f'Skip (exists): {video_name}')
                    continue
            try:
                raise NotImplementedError("Task 4 AM inference is disabled temporarily.")
                ans = query_task4(pipe, video_clip_fp, get_task4_AM_prompt())
                with open(task4_AM_result_csv_fp, 'a', newline='', encoding='utf-8') as f:
                    f.write(f'{video_name},{ans}\n')
            except Exception as e:
                print(f'Error (Task4 AM) {video_name}: {e}')
                with open(task4_AM_result_csv_fp, 'a', newline='', encoding='utf-8') as f:
                    f.write(f'{video_name},N/A\n')

    # Task 4L + 5
    t5_rng = validate_videos_range(task5_clip_fps, task5_videos_range)
    print(f'Task 4L+5 range: {t5_rng[0]} to {t5_rng[1]}')
    print(len(task5_clip_fps[t5_rng[0]-1: t5_rng[1]]))
    print(len(task5_clip_fps))
    for video_clip_fp in tqdm(task5_clip_fps[t5_rng[0]-1: t5_rng[1]], desc='Processing Task 4L & 5'):
        video_name = os.path.basename(video_clip_fp)
        if not os.path.exists(task4L_5_result_csv_fp):
            with open(task4L_5_result_csv_fp, 'w', newline='', encoding='utf-8') as f:
                f.write('video_name,start_time,end_time,onset_body_part\n')
        with open(task4L_5_result_csv_fp, 'r', encoding='utf-8') as f:
            if video_name in f.read():
                print(f'Skip (exists): {video_name}')
                continue
        try:
            raise NotImplementedError("Task 4L & 5 inference is disabled temporarily.")
            onset_body_part = query_task4_L(pipe, video_clip_fp, get_task4_L_prompt())
            start_time, end_time = query_task5(pipe, video_clip_fp)
            with open(task4L_5_result_csv_fp, 'a', newline='', encoding='utf-8') as f:
                f.write(f'{video_name},{start_time},{end_time},{onset_body_part}\n')
        except Exception as e:
            print(f'Error (Task4L+5) {video_name}: {e}')
            with open(task4L_5_result_csv_fp, 'a', newline='', encoding='utf-8') as f:
                f.write(f'{video_name},N/A,N/A,N/A\n')

    print(f'Task 4L+5 complete -> {task4L_5_result_csv_fp}')


if __name__ == '__main__':
    args = parse_arguments()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print('Starting InternVL3.5 multi-task (decord + IMAGE_TOKEN base64)...')
    print(f'GPU: {args.gpu}')
    print(f'Model: {args.model_name}')
    print(f'Output dir: {args.output_dir}')
    print(f'Max frames: {MAX_FRAMES}')
    print(f'FPS: {FPS}')
    print('-' * 60)

    main()
