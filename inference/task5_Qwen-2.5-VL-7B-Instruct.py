# import torch
# from PIL import Image
# from transformers import AutoModel, AutoTokenizer
# from torchvision.io import read_video    # pip install torchvision

import os
import json
from tqdm import tqdm
import re
import argparse

# default_model_cache_dir = os.path.join(os.path.dirname(__file__), 'model_cache')
# default_output_dir = os.path.join(os.path.dirname(__file__), 'output')
default_model_cache_dir = '/mnt/SSD3/tengyou/model_cache'
default_output_dir = '/mnt/SSD3/tengyou/output'

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
                       default='/mnt/SSD3/tengyou/seizure_videos/segments/all_dataset-backup',
                       help='Directory containing seizure video files')
    # cache directory
    parser.add_argument('--cache_dir', type=str, default=default_model_cache_dir,
                       help='Directory for model cache (default: ' + default_model_cache_dir + ')')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, default=default_output_dir,
                       help='Directory for output (default: ' + default_output_dir + ')')
    
    # Video range settings
    parser.add_argument('--videos_range', type=str, default='1-5',
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
    
    # Calculate timestamps for selected frames and round to nearest second
    timestamps = np.array([round(idx / fps) for idx in indices])

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
        # First try to extract JSON from the output if it's wrapped in text
        json_match = re.search(r'\{.*\}', vlm_output, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
        else:
            # Try to parse the entire output as JSON
            parsed = json.loads(vlm_output)
        
        # Check if it has the expected structure
        if isinstance(parsed, dict) and 'start_time' in parsed and 'end_time' in parsed:
            return parsed
        else:
            print(f"JSON parsed but missing required fields. Got: {parsed}")
            return {'start_time': 'N/A', 'end_time': 'N/A'}
            
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Failed to parse JSON: {vlm_output}")
        print(f"Error: {e}")
        # Return default values if parsing fails
        return {'start_time': 'N/A', 'end_time': 'N/A'}

def format_time(raw_output, offset):
    """
    Convert raw time string to 'MM:SS' after adding offset (in seconds).
    If raw_output == "N/A" or parsing fails, return "N/A".
    Handles various response formats from the model.
    Strips decimal places from seconds to ensure clean MM:SS format.
    """
    raw_output = raw_output.strip()
    
    # Check for N/A first
    if raw_output.upper() == "N/A":
        return "N/A"
    
    # Try to extract timestamp from various formats
    timestamp = None
    
    # Pattern 1: Look for [MM:SS] format
    bracket_match = re.search(r'\[(.*?)\]', raw_output)
    if bracket_match:
        timestamp = bracket_match.group(1)
    
    # Pattern 2: Look for MM:SS format directly (including potential decimal seconds)
    if not timestamp:
        time_match = re.search(r'(\d{1,2}:\d{2}(?:\.\d+)?)', raw_output)
        if time_match:
            timestamp = time_match.group(1)
    
    # Pattern 3: Look for "at MM:SS" or "MM:SS" in natural language (including potential decimal seconds)
    if not timestamp:
        time_match = re.search(r'at\s+(\d{1,2}:\d{2}(?:\.\d+)?)', raw_output, re.IGNORECASE)
        if time_match:
            timestamp = time_match.group(1)
    
    # If no timestamp found, return N/A
    if not timestamp:
        return "N/A"
    
    try:
        # Split timestamp and handle potential decimal seconds
        time_parts = timestamp.split(":")
        minutes = int(time_parts[0])
        # Remove decimal part from seconds if present
        seconds = int(float(time_parts[1]))
        
        total_seconds = minutes * 60 + seconds + int(offset)

        if total_seconds < 0:  # don't allow negative times
            return "00:00"

        # Round to nearest second to avoid decimal places
        total_seconds = round(total_seconds)
        minutes, seconds = divmod(total_seconds, 60)
        return f"{minutes:02}:{seconds:02}"
    except (ValueError, AttributeError):
        return "N/A"

def get_duration(start_time, end_time):
    """
    Compute duration in seconds given start and end times in 'MM:SS' or 'N/A'.
    Returns None if either time is 'N/A' or invalid.
    """
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
        return "N/A"  # guard against inverted times

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
    # print("raw output_text:", output_text)

    return output_text

# ================== Utility functions ==================

def AskEventTime(video_clip_fp):
    duration_prompt = "This is a seizure clip. Provide the start time and end time of the seizure event if it is present in this video clip. If the seizure has already started at the beginning of the video, use \"N/A\" for the start time. If the seizure has not ended by the end of the video, use \"N/A\" for the end time. Please return the result in the following JSON format: { start_time: MM:SS or N/A, end_time: MM:SS or N/A }"
    
    
    time_resp = inference(model, video_clip_fp, duration_prompt)
    time_resp = parse_json(time_resp)
    temp_start_time = time_resp['start_time']
    temp_end_time = time_resp['end_time']
    
    segment_idx = int(video_clip_fp.split('_segment_')[-1].split('.mp4')[0])
    
    time_offset = segment_idx * 25
    start_time = format_time(temp_start_time, time_offset)
    end_time = format_time(temp_end_time, time_offset)


   
    return start_time, end_time

# ================== Main function ==================

def main():
    # List all files in the directory to check existence quickly
    

    input_clip_fps = []
    for file in os.listdir(dataset_dir):
        if file.endswith('.mp4'):
            input_clip_fps.append(os.path.join(dataset_dir, file))

    # make sure videos_range is valid
    videos_range = args.videos_range.split('-')
    videos_range = [int(videos_range[0]), int(videos_range[1])]
    if len(videos_range) != 2:
        raise ValueError("videos_range must be a comma-separated string of two numbers")
    if int(videos_range[0]) > int(videos_range[1]):
        raise ValueError("videos_range[0] must be less than videos_range[1]")
    if int(videos_range[0]) < 1:
        videos_range[0] = 1
        # add warning
        print(f"Warning: videos_range[0] is less than 1, set to 1")
    if int(videos_range[1]) > len(input_clip_fps):
        videos_range[1] = len(input_clip_fps)
        # add warning
        print(f"Warning: videos_range[1] is greater than the number of videos, set to {len(input_clip_fps)}")
    
    if not os.path.exists(inf_result_csv_fp):
        with open(inf_result_csv_fp, 'w') as f:
            f.write("video_name,duration,start_time,end_time\n")

    for video_clip_fp in input_clip_fps[videos_range[0]-1 : videos_range[1]]:
        # skip if already in the CSV
        video_name = video_clip_fp.split('/')[-1]
        with open(inf_result_csv_fp, 'r') as f:
            if video_name in f.read():
                print(f"Video {video_name} already processed. Skipping.")
                continue

        try:
            start_time, end_time = AskEventTime(video_clip_fp)
            duration = get_duration(start_time, end_time)
            with open(inf_result_csv_fp, 'a') as f:
                f.write(f"{video_name},{duration},{start_time},{end_time}\n")
            print(f"Successfully processed {video_name}: start={start_time}, end={end_time}, duration={duration}")
        except Exception as e:
            print(f"Error processing video {video_name}: {e}")
            # Write error entry to CSV
            with open(inf_result_csv_fp, 'a') as f:
                f.write(f"{video_name},N/A,N/A,N/A\n")
            continue
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

