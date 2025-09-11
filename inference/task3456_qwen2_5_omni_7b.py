

# import torch
# from PIL import Image
# from transformers import AutoModel, AutoTokenizer
# from torchvision.io import read_video    # pip install torchvision

import os
import json
from torch.utils.data import dataset
from tqdm import tqdm

import argparse

LOG = False
# default_model_cache_dir = os.path.join(os.path.dirname(__file__), 'model_cache')
# default_output_dir = os.path.join(os.path.dirname(__file__), 'output')
default_model_cache_dir = '/mnt/SSD3/tengyou/model_cache'
default_output_dir = '/mnt/SSD3/tengyou/output'

def parse_arguments():
    parser = argparse.ArgumentParser(description='Seizure Video Feature Extraction using Qwen2.5-Omni')

    # GPU settings
    parser.add_argument('--gpu', type=str, default='3',
                       help='GPU device ID(s) to use (default: 0). Can be a single number or comma-separated numbers (e.g., 7 or 0,1,2)')

    # Model settings
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-Omni-7B',
                       help='Model name to use (default: Qwen/Qwen2.5-Omni-7B)')

    # Data settings
    parser.add_argument('--dataset_dir', type=str,
                       default=None,
                       help='Directory containing seizure video files')
    # cache directory
    parser.add_argument('--cache_dir', type=str, default=default_model_cache_dir,
                       help='Directory for model cache (default: ' + default_model_cache_dir + ')')

    # Output directory
    parser.add_argument('--output_dir', type=str, default=default_output_dir,
                       help='Directory for output (default: ' + default_output_dir + ')')

    # Video range settings
    parser.add_argument('--videos_range', type=str, default='1-2314',
                       help='Range of videos to process (e.g., "0,9" for first 10 videos, "10,19" for next 10 videos, etc.)')

    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()

# Set GPU environment variable
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Set the directories/paths from arguments
################################################################################################
model_name = args.model_name
task3_6_dataset_dir = os.path.join(args.dataset_dir, "task1_segments")
task4_HT_dataset_dir = os.path.join(args.dataset_dir, "task4_head_turning")
task4_AM_dataset_dir = os.path.join(args.dataset_dir, "task4_arm_movement")
task5_dataset_dir = os.path.join(args.dataset_dir, "task5_segment")

videos_range = (args.videos_range).split('-')
task3_6_videos_range = (args.videos_range).split('-')
task4_HT_videos_range = '1-129'.split('-')
task4_AM_videos_range = '1-112'.split('-')
task5_videos_range = (args.videos_range).split('-')

# inference files
inference_dir = args.output_dir
os.makedirs(inference_dir, exist_ok=True)

all_features = ['occur_during_sleep','blank_stare','close_eyes','eye_blinking',
            'tonic','clonic','arm_flexion','arm_straightening','figure4','oral_automatisms','limb_automatisms',
            'face_pulling','face_twitching','head_turning','asynchronous_movement','pelvic_thrusting',
            'arms_move_simultaneously','full_body_shaking',
            'ictal_vocalization', 'verbal_responsiveness',
            ]

# CSV file to read
task3_6_result_csv_fp = inference_dir + f'/Task3_6_{model_name.split("/")[-1]}_{task3_6_videos_range[0]}-{task3_6_videos_range[1]}.csv'
task4_HT_result_csv_fp = inference_dir + f'/Task4_HT_{model_name.split("/")[-1]}_{task4_HT_videos_range[0]}-{task4_HT_videos_range[1]}.csv'
task4_AM_result_csv_fp = inference_dir + f'/Task4_AM_{model_name.split("/")[-1]}_{task4_AM_videos_range[0]}-{task4_AM_videos_range[1]}.csv'
task4L_5_result_csv_fp = inference_dir + f'/Task4L_5_{model_name.split("/")[-1]}_{task5_videos_range[0]}-{task5_videos_range[1]}.csv'

################################################################################################
MAX_FRAMES = 60
FPS = 2
MAX_NEW_TOKENS = 2048
MAX_RETRIES = 10

import time
import traceback
import pandas as pd
import csv
import numpy as np

# Generate model_cache folder at the current directory
hf_cache_dir = os.path.join(args.cache_dir, 'huggingface')
modelscope_cache_dir = os.path.join(args.cache_dir, 'modelscope')
video_cache_dir = os.path.join(args.cache_dir, 'video_cache')
os.makedirs(hf_cache_dir, exist_ok=True)
os.makedirs(modelscope_cache_dir, exist_ok=True)
os.makedirs(video_cache_dir, exist_ok=True)

# Set environment variables for cache directories
os.environ['HF_HOME'] = hf_cache_dir
os.environ['MODELSCOPE_CACHE'] = modelscope_cache_dir

import torch
#from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from peft import PeftModel

# Load base model first
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto",
)
processor = Qwen2_5OmniProcessor.from_pretrained(model_name, cache_dir=hf_cache_dir)
import hashlib
import requests
from PIL import Image
import torchvision
from torchvision.io import read_video
import re
import pandas as pd

def download_video(url, dest_path):
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8096):
            f.write(chunk)
    print(f"Video downloaded to {dest_path}")

def get_video_frames(video_file_path, num_frames=128, cache_dir=video_cache_dir):
    video_hash = hashlib.md5(video_file_path.encode('utf-8')).hexdigest()
    frames_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_frames.npy')
    timestamps_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_timestamps.npy')

    if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
        frames = np.load(frames_cache_file)
        timestamps = np.load(timestamps_cache_file)
        return video_file_path, frames, timestamps

    video_tensor, audio_tensor, video_info = read_video(video_file_path, pts_unit='sec')
    total_frames = video_tensor.shape[0]
    fps = video_info['video_fps']

    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    selected_frames = video_tensor[indices]  # [num_frames, H, W, C]
    frames = selected_frames.numpy().astype(np.uint8)
    timestamps = np.array([idx / fps for idx in indices])

    np.save(frames_cache_file, frames)
    np.save(timestamps_cache_file, timestamps)
    return video_file_path, frames, timestamps

def create_image_grid(images, num_columns=8):
    pil_images = [Image.fromarray(image) for image in images]
    num_rows = (len(images) + num_columns - 1) // num_columns
    img_width, img_height = pil_images[0].size
    grid_width = num_columns * img_width
    grid_height = num_rows * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height))
    for idx, image in enumerate(pil_images):
        row_idx = idx // num_columns
        col_idx = idx % num_columns
        position = (col_idx * img_width, row_idx * img_height)
        grid_image.paste(image, position)
    return grid_image
import subprocess, os

def _extract_audio_to_wav(video_path: str, out_dir: str) -> str:
    base = os.path.splitext(os.path.basename(video_path))[0]
    wav_path = os.path.join(out_dir, f"{base}_24k_mono.wav")
    try:
        os.makedirs(out_dir, exist_ok=True)
        cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "24000", wav_path]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return wav_path if os.path.isfile(wav_path) else ""
    except Exception:
        return ""

def inference(model, video_path, query_prompt, max_new_tokens=None, max_pixels=602112, min_pixels=16 * 28 * 28):
    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS

    # 先尝试把音频轨抽出来（目录用你已有的 audio_cache_dir，或 video_cache_dir 也行）
    audio_path = _extract_audio_to_wav(video_path, audio_cache_dir if 'audio_cache_dir' in globals() else video_cache_dir)
    use_audio_in_video = False if audio_path else True

    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": video_path},
            *([{"type": "audio", "audio": audio_path}] if audio_path else []),
            {"type": "text", "text": query_prompt},
        ]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
    inputs = processor(
        text=[text],
        audio=audios, images=images, videos=videos,
        padding=True, return_tensors="pt",
        use_audio_in_video=use_audio_in_video,
    ).to(model.device)

    try:
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, use_audio_in_video=use_audio_in_video, return_audio=False)
    except TypeError:
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, use_audio_in_video=use_audio_in_video)

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

def get_task3_prompt():
    return '''
    Output the sequence of the any observed seizure symptoms of the patient in the video in chronological order.
    The symptoms are limited to head_turning, blank_stare, close_eyes, eye_blinking, face_pulling, face_twitching, tonic, clonic, arm_straightening, arm_flexion, figure4, oral_automatisms, limb_automatisms, asynchronous_movement, pelvic_thrusting, full_body_shaking, arms_move_simultaneously.
    If a symptom is not present in the video, it should not be included in the output.
    Example output: head_turning, arm_straightening, arm_flexion, tonic, clonic.
    Output only the seizure symptoms. Do not include any other text.
    '''
def get_task5_prompt():
    return '''
    This is a seizure clip. Provide the start time and end time of the seizure event if it is present in this video clip.
    If the seizure has already started at the beginning of the video, use "N/A" for the start time.
    If the seizure has not ended by the end of the video, use "N/A" for the end time.
    Return exactly ONE JSON object ONLY (no markdown code fences, no extra text), in the format:
    {"start_time": "MM:SS" or "N/A", "end_time": "MM:SS" or "N/A"}.
    Do not wrap the JSON in ``` and do not add any other text.
     '''

def get_task4_HT_prompt():
    return '''
    Does the patient's head turn to the patient's left or to the patient's right?
    Answer with \"left\" or \"right\" only. Do not include any extra text. Return exactly one word: left or right.
    '''

def get_task4_AM_prompt():
    return '''
    Which arm of the patient is moving in the video?
    Answer with \"left\" or \"right\" only. Do not include any extra text. Return exactly one word: left or right.
    '''

def get_task4_L_prompt():
    return '''
    Localize which body part shows the earliest visible seizure sign.
    Answer only with one of the following options: head, eyes, mouth, face, left arm, left leg, right arm, right leg, arms,legs,full body.
    '''

def get_task6_prompt():
    return """
    Generate a detailed report for this seizure video, describing the patient's observable actions, signs, and overall condition. The report must focus exclusively on the patient; do not include any descriptions of medical staff.
    For example: The patient is sleeping in bed. He lets out a loud groan and has versive head turn to the right. He has right upper extremity extension with left upper extremity flexion, followed by tonic-clonic activity. Later, the patient is unable to remember events preceding this seizure.
    Output the report as a cohesive paragraph in plain language. Do not include other content.
    """

def query_task3_6(video_clip_fp, log_file_fp):
    raw_output1 = inference(model, video_clip_fp, get_task3_prompt())
    clip_seq_text = '\"' + raw_output1 + '\"'
    raw_output2 = inference(model, video_clip_fp, get_task6_prompt())
    clip_report_text = '\"' + raw_output2 + '\"'
    if LOG:
        if not os.path.exists(log_file_fp):
            with open(log_file_fp, 'w') as f:
                f.write("video_fp,event_sequence,report\n")
        with open(log_file_fp, 'a') as f:
            f.write(f"{video_clip_fp},{clip_seq_text},{clip_report_text}\n")
    return clip_seq_text, clip_report_text

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
    if s in {'head', 'eyes', 'mouth', 'face', 'left arm', 'left leg', 'right arm', 'right leg', 'arms', 'legs'}:
        return s
    hits = re.findall(r'\b(head|eyes|mouth|face|left arm|left leg|right arm|right leg|arms|legs)\b', s)
    if hits:
        return hits[-1]
    return 'fail'

def query_task4(video_clip_fp, prompt):
    raw_answer = inference(model, video_clip_fp, prompt)
    ans = normalize_direction_task4(raw_answer)
    return ans

def query_task4_L(video_clip_fp, prompt):
    raw_answer = inference(model, video_clip_fp, prompt)
    ans = normalize_direction_task4_L(raw_answer)
    return ans

def parse_json_task5(vlm_output):
    try:
        s = vlm_output.strip()

        # 1) strip markdown code fences if present
        # remove all occurrences of ```json / ```JSON / ``` and ending ```
        s = s.replace('\r', '')
        s = re.sub(r'```(?:json|JSON)?\s*', '', s)
        s = re.sub(r'\s*```', '', s)

        # 2) extract the first {...} block
        m = re.search(r'\{.*?\}', s, re.DOTALL)
        json_str = m.group(0) if m else s

        # 3) normalize keys and times
        # ensure keys are quoted
        json_str = re.sub(r'(\bstart_time\b|\bend_time\b)\s*:', r'"\1":', json_str)
        # drop fractional seconds like 00:08.00 -> 00:08
        json_str = re.sub(r'(\d{1,2}:\d{2})\.\d+', r'\1', json_str)
        # ensure bare MM:SS are quoted
        json_str = re.sub(r'(?<!")(\b\d{1,2}:\d{2}\b)(?!")', r'"\1"', json_str)
        # normalize NA variants
        json_str = re.sub(r'("start_time"|"end_time")\s*:\s*(NA|N/?A|null|None)', r'\1: "N/A"', json_str, flags=re.IGNORECASE)

        # 4) try parse
        parsed = json.loads(json_str)
        if isinstance(parsed, dict) and 'start_time' in parsed and 'end_time' in parsed:
            return parsed
        else:
            print(f"JSON parsed but missing required fields. Got: {parsed}")
            return {'start_time': 'N/A', 'end_time': 'N/A'}
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Failed to parse JSON: {vlm_output}")
        print(f"Error: {e}")
        return {'start_time': 'N/A', 'end_time': 'N/A'}

def format_time_task5(raw_output):
    raw_output = raw_output.strip()
    if raw_output.upper() == "N/A":
        return "N/A"
    timestamp = None
    bracket_match = re.search(r'\[(.*?)\]', raw_output)
    if bracket_match:
        timestamp = bracket_match.group(1)
    if not timestamp:
        time_match = re.search(r'(\d{1,2}:\d{2}(?:\.\d+)?)', raw_output)
        if time_match:
            timestamp = time_match.group(1)
    if not timestamp:
        time_match = re.search(r'at\s+(\d{1,2}:\d{2}(?:\.\d+)?)', raw_output, re.IGNORECASE)
        if time_match:
            timestamp = time_match.group(1)
    if not timestamp:
        return "N/A"
    try:
        time_parts = timestamp.split(":")
        minutes = int(time_parts[0])
        seconds = int(float(time_parts[1]))
        total_seconds = minutes * 60 + seconds
        if total_seconds < 0:
            return "00:00"
        total_seconds = round(total_seconds)
        minutes, seconds = divmod(total_seconds, 60)
        return f"{minutes:02}:{seconds:02}"
    except (ValueError, AttributeError):
        return "N/A"

def query_task5(video_clip_fp):
    time_resp = inference(model, video_clip_fp, get_task5_prompt())
    time_resp = parse_json_task5(time_resp)
    temp_start_time = time_resp['start_time']
    temp_end_time = time_resp['end_time']
    start_time = format_time_task5(temp_start_time)
    end_time = format_time_task5(temp_end_time)
    return start_time, end_time

def get_duration_task5(start_time, end_time):
    def to_seconds(ts):
        if ts == "N/A":
            return "N/A"
        try:
            m, s = map(int, ts.split(":"))
            return m * 60 + s
        except ValueError:
            print(f"Invalid time format: {ts}")
            return "N/A"

    start_sec = to_seconds(start_time)
    end_sec = to_seconds(end_time)
    if start_sec == "N/A" or end_sec == "N/A":
        return "N/A"
    if end_sec < start_sec:
        return "N/A"
    return end_sec - start_sec

from typing import List
def validate_videos_range(clip_files:List[str], videos_range:List):
    videos_range = [int(videos_range[0]), int(videos_range[1])]
    if len(videos_range) != 2:
        raise ValueError("videos_range must be a comma-separated string of two numbers")
    if int(videos_range[0]) > int(videos_range[1]):
        raise ValueError("videos_range[0] must be less than videos_range[1]")
    if int(videos_range[0]) < 1:
        videos_range[0] = 1
        print(f"Warning: videos_range[0] is less than 1, set to 1")
    if int(videos_range[1]) > 2300:
        videos_range[1] = len(clip_files)
    if int(videos_range[1]) > len(clip_files):
        videos_range[1] = len(clip_files)
        print(f"Warning: videos_range[1] is greater than the number of videos, set to {len(clip_files)}")
    return videos_range

def get_fp_list(file_dir):
    fp_list = []
    input_videos_files = os.listdir(file_dir)
    input_videos_files = set(input_videos_files)
    input_videos_files = sorted(input_videos_files)
    for file in input_videos_files:
        if file.endswith('.mp4'):
            fp_list.append(os.path.join(file_dir, file))
    return fp_list

def main():
    global videos_range
    global task3_6_videos_range, task4_HT_videos_range, task4_AM_videos_range, task5_videos_range
    global task3_6_dataset_dir, task4_HT_dataset_dir, task4_AM_dataset_dir, task5_dataset_dir
    global task3_6_result_csv_fp, task4_HT_result_csv_fp, task4_AM_result_csv_fp, task4L_5_result_csv_fp

    task3_6_clip_fps = get_fp_list(task3_6_dataset_dir)
    task4_HT_clip_fps = get_fp_list(task4_HT_dataset_dir)
    task4_AM_clip_fps = get_fp_list(task4_AM_dataset_dir)
    task5_clip_fps = get_fp_list(task5_dataset_dir)

    # =============================================== task3 + task6 =============================================================== #
    task_3_6_videos_range = validate_videos_range(task3_6_clip_fps, task3_6_videos_range)
    for video_clip_fp in tqdm(task3_6_clip_fps[task_3_6_videos_range[0]-1 : task_3_6_videos_range[1]], desc="Processing Task 3 and Task 6"):
        video_clip_name = video_clip_fp.split('/')[-1]
        if not os.path.exists(task3_6_result_csv_fp):
            with open(task3_6_result_csv_fp, 'w') as f:
                f.write("video_name,event_sequence,report\n")
        with open(task3_6_result_csv_fp, 'r') as f:
            if video_clip_name in f.read():
                print(f"Video {video_clip_name} already processed. Skipping.")
                continue
        if LOG:
            log_dir = os.path.join(inference_dir, 'log')
            os.makedirs(log_dir, exist_ok=True)
            log_file_fp = os.path.join(log_dir, f"{video_clip_name}.csv")
        else:
            log_file_fp = None
        video_clip_fp = os.path.join(task3_6_dataset_dir, video_clip_name)
        video_event_seq_list, clip_report_text = query_task3_6(video_clip_fp, log_file_fp)
        with open(task3_6_result_csv_fp, 'a') as f:
            f.write(f"{video_clip_name},{video_event_seq_list},{clip_report_text}\n")
    print(f"Processing is complete. Results are in '{task3_6_result_csv_fp}'.")

    # =============================================== task4 =============================================================== #
    if int(videos_range[1]) > 2300:
        task4_HT_videos_range = validate_videos_range(task4_HT_clip_fps, task4_HT_videos_range)
        for video_clip_fp in tqdm(task4_HT_clip_fps[:], desc="Processing Task 4 HT"):
            video_name = video_clip_fp.split('/')[-1]
            if not os.path.exists(task4_HT_result_csv_fp):
                with open(task4_HT_result_csv_fp, 'w') as f:
                    f.write("video_name,head_turning_direction\n")
            with open(task4_HT_result_csv_fp, 'r') as f:
                if video_name in f.read():
                    print(f"Video {video_name} already processed. Skipping.")
                    continue
            try:
                HT_ans = query_task4(video_clip_fp, get_task4_HT_prompt())
                HT_ans = normalize_direction_task4(HT_ans)
                with open(task4_HT_result_csv_fp, 'a') as f:
                    f.write(f"{video_name},{HT_ans}\n")
            except Exception as e:
                print(f"Error processing video {video_name} in Task 4 HT: {e}")
                with open(task4_HT_result_csv_fp, 'a') as f:
                    f.write(f"{video_name},N/A\n")
                continue

        task_4_AM_video_range = validate_videos_range(task4_AM_clip_fps, task4_AM_videos_range)
        for video_clip_fp in tqdm(task4_AM_clip_fps[:], desc="Processing Task 4 AM"):
            video_name = video_clip_fp.split('/')[-1]
            if not os.path.exists(task4_AM_result_csv_fp):
                with open(task4_AM_result_csv_fp, 'w') as f:
                    f.write("video_name,arm_movement_direction\n")
            with open(task4_AM_result_csv_fp, 'r') as f:
                if video_name in f.read():
                    print(f"Video {video_name} already processed. Skipping.")
                    continue
            try:
                AM_ans = query_task4(video_clip_fp, get_task4_AM_prompt())
                AM_ans = normalize_direction_task4(AM_ans)
                with open(task4_AM_result_csv_fp, 'a') as f:
                    f.write(f"{video_name},{AM_ans}\n")
            except Exception as e:
                print(f"Error processing video {video_name} in Task 4 AM: {e}")
                with open(task4_AM_result_csv_fp, 'a') as f:
                    f.write(f"{video_name},N/A\n")
                continue

    # =============================================== task5 =============================================================== #
    task_5_videos_range = validate_videos_range(task5_clip_fps, task5_videos_range)
    for video_clip_fp in tqdm(task5_clip_fps[task_5_videos_range[0]-1 : task_5_videos_range[1]], desc="Processing Task 5"):
        video_name = video_clip_fp.split('/')[-1]
        if not os.path.exists(task4L_5_result_csv_fp):
            with open(task4L_5_result_csv_fp, 'w') as f:
                f.write("video_name,start_time,end_time,onset_body_part\n")
        with open(task4L_5_result_csv_fp, 'r') as f:
            if video_name in f.read():
                print(f"Video {video_name} already processed. Skipping.")
                continue
        try:
            onset_body_part = query_task4_L(video_clip_fp, get_task4_L_prompt())
            onset_body_part = normalize_direction_task4_L(onset_body_part)
            start_time, end_time = query_task5(video_clip_fp)
            with open(task4L_5_result_csv_fp, 'a') as f:
                f.write(f"{video_name},{start_time},{end_time},{onset_body_part}\n")
            print(f"Successfully processed {video_name}: start={start_time}, end={end_time}, onset_body_part={onset_body_part}")
        except Exception as e:
            print(f"Error processing video Task 4L & 5 {video_name}: {e}")
            with open(task4L_5_result_csv_fp, 'a') as f:
                f.write(f"{video_name},N/A,N/A,N/A\n")
            continue
    print(f"Processing is complete. Results are in '{task4L_5_result_csv_fp}'.")

if __name__ == "__main__":
    print(f"Starting seizure video feature extraction...")
    print(f"GPU: {args.gpu}")
    print(f"Model: {args.model_name}")
    print(f"Output: {inference_dir}")
    print(f"Max frames: {MAX_FRAMES}")
    print(f"FPS: {FPS}")
    print("-" * 50)
    main()
