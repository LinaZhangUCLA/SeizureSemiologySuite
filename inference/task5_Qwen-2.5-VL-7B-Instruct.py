# import torch
# from PIL import Image
# from transformers import AutoModel, AutoTokenizer
# from torchvision.io import read_video    # pip install torchvision

import os
import json
from tqdm import tqdm

import argparse

default_model_cache_dir = os.path.join(os.path.dirname(__file__), 'model_cache')
default_output_dir = os.path.join(os.path.dirname(__file__), 'output')
# default_model_cache_dir = '/mnt/SSD3/tengyou/model_cache'
# default_output_dir = '/mnt/SSD3/tengyou/output'

def parse_arguments():
    parser = argparse.ArgumentParser(description='Seizure Video Feature Extraction using Qwen2.5-VL')
    
    # GPU settings
    parser.add_argument('--gpu', type=str, default='0', 
                       help='GPU device ID(s) to use (default: 3). Can be a single number or comma-separated numbers (e.g., 7 or 0,1,2)')
    
    # Model settings
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct',
                       help='Model name to use (default: Qwen/Qwen2.5-VL-7B-Instruct)')
    
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
    parser.add_argument('--videos_range', type=str, default='0-4',
                       help='Range of videos to process (e.g., "0,9" for first 10 videos, "10,19" for next 10 videos, etc.)')
   
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()

# Set GPU environment variable
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Set the directories/paths from arguments
################################################################################################
model_name = args.model_name
dataset_dir = args.dataset_dir
videos_range = args.videos_range

# inference files
inference_dir = args.output_dir
os.makedirs(inference_dir, exist_ok=True)

all_features = ['occur_during_sleep','blank_stare','close_eyes','eye_blinking',
            'tonic','clonic','arm_flexion','arm_straightening','figure4','oral_automatisms','limb_automatisms',
            'face_pulling','face_twitching','head_turning','asynchronous_movement','pelvic_thrusting',
            'arms_move_simultaneously','full_body_shaking', 
            # 'ictal_vocalization', 'verbal_responsiveness',
            ]

# CSV file to read
inf_result_csv_fp = inference_dir + f'/Task5_{model_name.split("/")[-1]}_{videos_range}.csv'

# Common video resolutions for target_size parameter:
# 1080p: (1920, 1080)
# 720p:  (1280, 720) 
# 480p:  (854, 480)
# 360p:  (640, 360)
# 240p:  (426, 240)
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
# from lmdeploy import pipeline, TurbomindEngineConfig
# from lmdeploy import pipeline, GenerationConfig
# from torchvision.io import read_video
# from lmdeploy.vl.constants import IMAGE_TOKEN
# from lmdeploy.vl.utils import encode_image_base64
# from PIL import Image

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
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel

# Load base model first
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2",
    device_map="auto"
    
)

# Move model to GPU
# model = model.to('cuda')


# Load processor from the base model name, not the checkpoint
processor = AutoProcessor.from_pretrained(model_name, cache_dir=hf_cache_dir)


import os
import hashlib
import requests

#from IPython.display import Markdown, display
import numpy as np
from PIL import Image
import torchvision
from torchvision.io import read_video


def download_video(url, dest_path):
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8096):
            f.write(chunk)
    print(f"Video downloaded to {dest_path}")


def get_video_frames(video_file_path, num_frames=128, cache_dir=video_cache_dir):
    # os.makedirs(cache_dir, exist_ok=True)

    video_hash = hashlib.md5(video_file_path.encode('utf-8')).hexdigest()


    frames_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_frames.npy')
    timestamps_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_timestamps.npy')

    if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
        frames = np.load(frames_cache_file)
        timestamps = np.load(timestamps_cache_file)
        return video_file_path, frames, timestamps

    # Read video using torchvision
    video_tensor, audio_tensor, video_info = read_video(video_file_path, pts_unit='sec')
    total_frames = video_tensor.shape[0]
    fps = video_info['video_fps']
    
    # print("total_frames : ", total_frames)

    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    
    # Extract selected frames
    selected_frames = video_tensor[indices]  # Shape: [num_frames, H, W, C]
    
    # Convert to numpy array and ensure uint8 format
    frames = selected_frames.numpy().astype(np.uint8)
    
    # Calculate timestamps for selected frames
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

def parse_json(vlm_output):
    # Parse the JSON output from the VLM
    try:
        features = json.loads(vlm_output)
        assert isinstance(features, list)
    except (json.JSONDecodeError, AssertionError):
        print(f"Failed to parse JSON: {vlm_output}")
        features = []
    return features

def format_time(raw_output, offset):
    """
    Convert raw time string 'MM:SS' to 'MM:SS' after adding offset (in seconds).
    If raw_output == "N/A" or parsing fails, return "N/A".
    """
    raw_output = raw_output.strip()
    if raw_output.upper() == "N/A":
        return "N/A"
    try:
        minutes, seconds = map(int, raw_output.split(":"))
        total_seconds = minutes * 60 + seconds + int(offset)

        if total_seconds < 0:  # don't allow negative times
            return "00:00"

        minutes, seconds = divmod(total_seconds, 60)
        return f"{minutes:02}:{seconds:02}"
    except (ValueError, AttributeError):
        raise ValueError(f"Invalid time format: {raw_output}")

def get_duration(start_time, end_time):
    """
    Compute duration in seconds given start and end times in 'MM:SS' or 'N/A'.
    Returns None if either time is 'N/A' or invalid.
    """
    def to_seconds(ts):
        if ts == "N/A":
            return None
        try:
            m, s = map(int, ts.split(":"))
            return m * 60 + s
        except ValueError:
            return None

    start_sec = to_seconds(start_time)
    end_sec = to_seconds(end_time)

    if start_sec is None or end_sec is None:
        return None
    if end_sec < start_sec:
        return None  # guard against inverted times

    return end_sec - start_sec

def inference(model, video_path, query_prompt, max_new_tokens=None, max_pixels=602112, min_pixels=16 * 28 * 28):
    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS
    messages = [
        # {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {
                "type": "video",
                "video": video_path,
                "max_pixels": max_pixels,
                "min_pixels": min_pixels,
                "total_pixels": max_pixels * MAX_FRAMES,
                "fps": FPS,
            },
            {"type": "text", "text": query_prompt},
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    # print("video input:", video_inputs[0].shape)
    num_frames, _, resized_height, resized_width = video_inputs[0].shape
    # print("num of video tokens:", int(num_frames / 2 * resized_height / 28 * resized_width / 28))
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    output_text = output_text[0]

    return output_text

# ================== Utility functions ==================

def AskEventTime(video_fn, video_clip_dir):
    video_fn = video_fn.split('/')[-1].split('.mp4')[0]

    # get all clips
    video_clips = []
    for file in os.listdir(video_clip_dir):
        if video_fn in file and file.endswith('.mp4'):
            video_fp = os.path.join(video_clip_dir, file)
            video_clips.append(video_fp)

    all_start_times = []
    all_end_times = []
    start_time_prompt = "Does the event start in this clip? If yes, give me the start timestamp in MM:SS, else return N/A."
    end_time_prompt = "Does the event end in this clip? If yes, give me the end timestamp in MM:SS, else return N/A."
    for idx, clip_fp in tqdm(enumerate(video_clips), total=len(video_clips), desc=f"Processing clips of video {video_fn}"):
        temp_start_time = inference(model, clip_fp, start_time_prompt)
        temp_end_time = inference(model, clip_fp, end_time_prompt)

        time_offset = idx * 25
        start_time = format_time(temp_start_time, time_offset)
        end_time = format_time(temp_end_time, time_offset)

        all_start_times.append(start_time)
        all_end_times.append(end_time)

    # convert to a string
    global_start_time = min([t for t in all_start_times if t != "N/A"], default="N/A")
    global_end_time = min([t for t in all_end_times if t != "N/A"], default="N/A")
    return global_start_time, global_end_time

# ================== Main function ==================

def main():
    # List all files in the directory to check existence quickly
    input_clip_files = os.listdir(dataset_dir)

    input_video_files = []
    for file in input_clip_files:
        video_fn = file.split('.mp4')[0]
        video_fn = '_'.join(video_fn.split('_')[:-1])+'.mp4'
        input_video_files.append(video_fn)
    input_videos_files = list(set(input_video_files))

    if not os.path.exists(inf_result_csv_fp):
        with open(inf_result_csv_fp, 'w') as f:
            f.write("video_name,duration,start_time,end_time\n")

    for video_name_idx, video_name in enumerate(input_videos_files):
        print(f"Processing video {video_name_idx + 1}/{len(input_videos_files)}: {video_name}")
        # skip if already in the CSV
        with open(inf_result_csv_fp, 'r') as f:
            if video_name in f.read():
                print(f"Video {video_name} already processed. Skipping.")
                continue

        video_clip_dir = os.path.join(inference_dir, video_name.split('.mp4')[0])
        start_time, end_time = AskEventTime(video_name, video_clip_dir)
        duration = get_duration(start_time, end_time)
        with open(inf_result_csv_fp, 'a') as f:
            f.write(f"{video_name},{duration},{start_time},{end_time}\n")
    print(f"Processing is complete. Results are in '{inf_result_csv_fp}'.")

if __name__ == "__main__":
    print(f"Starting seizure video feature extraction...")
    print(f"GPU: {args.gpu}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Output: {inference_dir}")
    print(f"Videos range: {args.videos_range}")
    print(f"Max frames: {MAX_FRAMES}")
    print(f"FPS: {FPS}")
    print("-" * 50)
    
    main()

