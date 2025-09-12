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
    parser = argparse.ArgumentParser(description='Seizure Video Feature Extraction using Qwen2.5-VL')
    
    # GPU settings
    parser.add_argument('--gpu', type=str, default='3', 
                       help='GPU device ID(s) to use (default: 0). Can be a single number or comma-separated numbers (e.g., 7 or 0,1,2)')

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
    parser.add_argument('--videos_range', type=str, default='1-2314',
                       help='Range of videos to process (e.g., "0,9" for first 10 videos, "10,19" for next 10 videos, etc.)')                   
    
    # # Data settings
    # parser.add_argument('--task3_6_dataset_dir', type=str, 
    #                    default='/mnt/SSD3/tengyou/seizure_videos/segments/all_dataset',
    #                    help='Directory containing seizure video files')
    # parser.add_argument('--task4_HT_dataset_dir', type=str, 
    #                    default='/mnt/SSD3/xinyi/benchmark/video_segment/clips_head_turning',
    #                    help='Directory containing seizure video files')
    # parser.add_argument('--task4_AM_dataset_dir', type=str, 
    #                    default='/mnt/SSD3/xinyi/benchmark/video_segment/clips_arm_movement',
    #                    help='Directory containing seizure video files')
    # parser.add_argument('--task5_dataset_dir', type=str, 
    #                    default='/mnt/SSD3/tengyou/benchmark_tasks/task5/segments',
    #                 # default='/mnt/SSD3/tengyou/seizure_videos/segments/all_dataset',
    #                    help='Directory containing seizure video files')
    # # cache directory
    # parser.add_argument('--cache_dir', type=str, default=default_model_cache_dir,
    #                    help='Directory for model cache (default: ' + default_model_cache_dir + ')')
    
    # # Output directory
    # parser.add_argument('--output_dir', type=str, default=default_output_dir,
    #                    help='Directory for output (default: ' + default_output_dir + ')')
    
    # # Video range settings
    # parser.add_argument('--task3_6_videos_range', type=str, default='1-5',
    #                    help='Range of videos to process (e.g., "1-100" for first 100 videos)')
    # parser.add_argument('--task4_HT_videos_range', type=str, default='1-5',
    #                    help='Range of videos to process (e.g., "1-100" for first 100 videos)')
    # parser.add_argument('--task4_AM_videos_range', type=str, default='1-5',
    #                    help='Range of videos to process (e.g., "1-100" for first 100 videos)')
    # parser.add_argument('--task5_videos_range', type=str, default='1-5',
    #                    help='Range of videos to process (e.g., "1-100" for first 100 videos)')
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
            # 'ictal_vocalization', 'verbal_responsiveness',
            ]

# CSV file to read
task3_6_result_csv_fp = inference_dir + f'/Task3_6_{model_name.split("/")[-1]}_{task3_6_videos_range[0]}-{task3_6_videos_range[1]}.csv'
task4_HT_result_csv_fp = inference_dir + f'/Task4_HT_{model_name.split("/")[-1]}_{task4_HT_videos_range[0]}-{task4_HT_videos_range[1]}.csv'
task4_AM_result_csv_fp = inference_dir + f'/Task4_AM_{model_name.split("/")[-1]}_{task4_AM_videos_range[0]}-{task4_AM_videos_range[1]}.csv'
task4L_5_result_csv_fp = inference_dir + f'/Task4L_5_{model_name.split("/")[-1]}_{task5_videos_range[0]}-{task5_videos_range[1]}.csv'


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
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     model_name, 
#     torch_dtype=torch.bfloat16, 
#     attn_implementation="flash_attention_2",
#     device_map="auto"
    
# )

# Move model to GPU
# model = model.to('cuda')


# Load processor from the base model name, not the checkpoint
# processor = AutoProcessor.from_pretrained(model_name, cache_dir=hf_cache_dir)

model, processor = None, None
import os
import hashlib
import requests

#from IPython.display import Markdown, display
import numpy as np
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

def inference(model, video_path, query_prompt, max_new_tokens=None, max_pixels=602112, min_pixels=16 * 28 * 28):
    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS
    messages = [
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
    raw_output = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # print("raw_output:", raw_output[0])
    
    return raw_output[0]

# ================== Paths and file names ==================


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
    If the seizure has already started at the beginning of the video, use \"N/A\" for the start time. 
    If the seizure has not ended by the end of the video, use \"N/A\" for the end time. 
    Please return the result in the following JSON format: { start_time: MM:SS or N/A, end_time: MM:SS or N/A }. Do not include any other text. 
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
    # return'''
    # Generate a detailed report for this seizure video, and the symptoms are limited to head_turning, blank_stare, close_eyes, eye_blinking, face_pulling, face_twitching, tonic, clonic, arm_straightening, arm_flexion, figure4, oral_automatisms, limb_automatisms, asynchronous_movement, pelvic_thrusting, full_body_shaking, arms_move_simultaneously, verbal_responsiveness, ictal_vocalization.
    # Only focus on the patient. Do not include any description for the medical staff.
    # For example: The patient is sleeping in bed. He lets out a loud groan and has versive head turn to the right. He has right upper extremity extension with left upper extremity flexion, followed by tonic-clonic activity. Later, the patient is unable to remember events preceding this seizure.
    # Output the report in several sentences, plain language. Do not include other content. 
    # '''
    return """
    Generate a detailed report for this seizure video, describing the patient's observable actions, signs, and overall condition. The report must focus exclusively on the patient; do not include any descriptions of medical staff.
    For example: The patient is sleeping in bed. He lets out a loud groan and has versive head turn to the right. He has right upper extremity extension with left upper extremity flexion, followed by tonic-clonic activity. Later, the patient is unable to remember events preceding this seizure.
    Output the report as a cohesive paragraph in plain language. Do not include other content.
    """

# ================== Utility functions ==================

def query_task3_6(video_clip_fp, log_file_fp):
    # task3 + task6 query
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
    """Map free-text variants to 'left' or 'right' (fallback 'fail')."""
    if ans is None:
        return 'fail'
    s = str(ans).strip().lower()
    # exact single-word
    if s in {'left', 'right'}:
        return s
    # find any occurrence; take the last one to be robust against echoing the question
    hits = re.findall(r'\b(left|right)\b', s)
    if hits:
        return hits[-1]
    # abbreviations
    if re.fullmatch(r'[lr]', s):
        return 'left' if s == 'l' else 'right'
    return 'fail'

def normalize_direction_task4_L(ans: str) -> str:
    """Map free-text variants to 'left' or 'right' (fallback 'fail')."""
    if ans is None:
        return 'fail'
    s = str(ans).strip().lower()
    # exact single-word
    if s in {'head', 'eyes', 'mouth', 'face', 'left arm', 'left leg', 'right arm', 'right leg', 'arms', 'legs'}:
        return s
    # find any occurrence; take the last one to be robust against echoing the question
    hits = re.findall(r'\b(head|eyes|mouth|face|left arm|left leg|right arm|right leg|arms|legs)\b', s)
    if hits:
        return hits[-1]
    return 'fail'

def query_task4(video_clip_fp, prompt):
    """
    Run a single left/right prompt and parse the answer (one word).
    Optionally log (file_name, prompt, answer) to per-video CSV.
    """
    raw_answer = inference(model, video_clip_fp, prompt)
    ans = normalize_direction_task4(raw_answer)
    return ans

def query_task4_L(video_clip_fp, prompt):
    raw_answer = inference(model, video_clip_fp, prompt)
    ans = normalize_direction_task4_L(raw_answer)
    return ans

def parse_json_task5(vlm_output):
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

def format_time_task5(raw_output):
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
        
        total_seconds = minutes * 60 + seconds

        if total_seconds < 0:  # don't allow negative times
            return "00:00"

        # Round to nearest second to avoid decimal places
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

# ================== Main function ==================

from typing import List
def validate_videos_range(clip_files:List[str], videos_range:List):
    # videos_range = args.videos_range.split('-')
    videos_range = [int(videos_range[0]), int(videos_range[1])]
    if len(videos_range) != 2:
        raise ValueError("videos_range must be a comma-separated string of two numbers")
    if int(videos_range[0]) > int(videos_range[1]):
        raise ValueError("videos_range[0] must be less than videos_range[1]")
    if int(videos_range[0]) < 1:
        videos_range[0] = 1
        # add warning
        print(f"Warning: videos_range[0] is less than 1, set to 1")
    if int(videos_range[1]) > 2300:
        videos_range[1] = len(clip_files)    
    if int(videos_range[1]) > len(clip_files):
        videos_range[1] = len(clip_files)
        # add warning
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
    
    # make sure videos_range is valid
    task_3_6_videos_range = validate_videos_range(task3_6_clip_fps, task3_6_videos_range)
    for video_clip_fp in tqdm(task3_6_clip_fps[task_3_6_videos_range[0]-1 : task_3_6_videos_range[1]], desc="Processing Task 3 and Task 6"):
        # skip if already in the CSV
        video_clip_name = video_clip_fp.split('/')[-1]
        if not os.path.exists(task3_6_result_csv_fp):
            with open(task3_6_result_csv_fp, 'w') as f:
                f.write("video_name,event_sequence,report\n")
                
        with open(task3_6_result_csv_fp, 'r') as f:
            if video_clip_name in f.read():
                print(f"Video {video_clip_name} already processed. Skipping.")
                raise NotImplementedError("find repeaete.")
                continue
        if LOG:
            log_dir = os.path.join(inference_dir, 'log')
            os.makedirs(log_dir, exist_ok=True)
            log_file_fp = os.path.join(log_dir, f"{video_clip_name}.csv")
        else:
            log_file_fp = None
        video_clip_fp = os.path.join(task3_6_dataset_dir, video_clip_name)
        # video_event_seq_list, clip_report_text = query_task3_6(video_clip_fp, log_file_fp)
        video_event_seq_list, clip_report_text = '',''
        with open(task3_6_result_csv_fp, 'a') as f:
            f.write(f"{video_clip_name},{video_event_seq_list},{clip_report_text}\n")
    print(f"Processing is complete. Results are in '{task3_6_result_csv_fp}'.")
    
    # =============================================== task4 =============================================================== #
    # task4 part1
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
                    raise NotImplementedError("find repeaete.")
                    continue
            
            try:
                raise NotImplementedError("Task 4L & 5 inference is disabled temporarily.")
                HT_ans = query_task4(video_clip_fp, get_task4_HT_prompt())
                HT_ans = normalize_direction_task4(HT_ans)
                with open(task4_HT_result_csv_fp, 'a') as f:
                    f.write(f"{video_name},{HT_ans}\n")
            except Exception as e:
                #print(f"Error processing video {video_name} in Task 4 HT: {e}")
                # Write error entry to CSV
                with open(task4_HT_result_csv_fp, 'a') as f:
                    f.write(f"{video_name},N/A\n")
                continue
    
        # task4 part2
        task_4_AM_video_range = validate_videos_range(task4_AM_clip_fps, task4_AM_videos_range)
        for video_clip_fp in tqdm(task4_AM_clip_fps[:], desc="Processing Task 4 AM"):
            video_name = video_clip_fp.split('/')[-1]
            if not os.path.exists(task4_AM_result_csv_fp):
                with open(task4_AM_result_csv_fp, 'w') as f:
                    f.write("video_name,arm_movement_direction\n")
            with open(task4_AM_result_csv_fp, 'r') as f:
                if video_name in f.read():
                    print(f"Video {video_name} already processed. Skipping.")
                    raise NotImplementedError("find repeaete.")
                    continue
            
            try:
                raise NotImplementedError("Task 4L & 5 inference is disabled temporarily.")
                AM_ans = query_task4(video_clip_fp, get_task4_AM_prompt())
                AM_ans = normalize_direction_task4(AM_ans)
                with open(task4_AM_result_csv_fp, 'a') as f:
                    f.write(f"{video_name},{AM_ans}\n")
            except Exception as e:
                #print(f"Error processing video {video_name} in Task 4 AM: {e}")
                # Write error entry to CSV
                with open(task4_AM_result_csv_fp, 'a') as f:
                    f.write(f"{video_name},N/A\n")
                continue
    

    # =============================================== task5 =============================================================== #
    task_5_videos_range = validate_videos_range(task5_clip_fps, task5_videos_range)
    for video_clip_fp in tqdm(task5_clip_fps[task_5_videos_range[0]-1 : task_5_videos_range[1]], desc="Processing Task 5"):
        # skip if already in the CSV
        video_name = video_clip_fp.split('/')[-1]
        
        if not os.path.exists(task4L_5_result_csv_fp):
            with open(task4L_5_result_csv_fp, 'w') as f:
                f.write("video_name,start_time,end_time,onset_body_part\n")

        with open(task4L_5_result_csv_fp, 'r') as f:
            if video_name in f.read():
                print(f"Video {video_name} already processed. Skipping.")
                raise NotImplementedError("find repeaete.")
                continue

        try:
            raise NotImplementedError("Task 4L & 5 inference is disabled temporarily.")
            onset_body_part = query_task4_L(video_clip_fp, get_task4_L_prompt())
            onset_body_part = normalize_direction_task4_L(onset_body_part)
            start_time, end_time = query_task5(video_clip_fp)
            with open(task4L_5_result_csv_fp, 'a') as f:
                f.write(f"{video_name},{start_time},{end_time},{onset_body_part}\n")
            print(f"Successfully processed {video_name}: start={start_time}, end={end_time}, onset_body_part={onset_body_part}")
        except Exception as e:
            #print(f"Error processing video Task 4L & 5 {video_name}: {e}")
            # Write error entry to CSV
            with open(task4L_5_result_csv_fp, 'a') as f:
                f.write(f"{video_name},N/A,N/A,N/A\n")
            continue
    print(f"Processing is complete. Results are in '{task4L_5_result_csv_fp}'.")

if __name__ == "__main__":
    print(f"Starting seizure video feature extraction...")
    print(f"GPU: {args.gpu}")
    print(f"Model: {args.model_name}")
    # print(f"Task 3+6 dataset: {args.task3_6_dataset_dir}")
    # print(f"Task 4 HT dataset: {args.task4_HT_dataset_dir}")
    # print(f"Task 4 AM dataset: {args.task4_AM_dataset_dir}")
    # print(f"Task 5 dataset: {args.task5_dataset_dir}")
    print(f"Output: {inference_dir}")
    # print(f"Task 3+6 videos range: {args.task3_6_videos_range}")
    # print(f"Task 4 HT videos range: {args.task4_HT_videos_range}")
    # print(f"Task 4 AM videos range: {args.task4_AM_videos_range}")
    # print(f"Task 4L+5 videos range: {args.task5_videos_range}")
    print(f"Max frames: {MAX_FRAMES}")
    print(f"FPS: {FPS}")
    print("-" * 50)
    
    main()

