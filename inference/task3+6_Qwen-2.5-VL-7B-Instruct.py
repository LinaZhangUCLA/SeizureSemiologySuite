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
    parser.add_argument('--gpu', type=str, default='0', 
                       help='GPU device ID(s) to use (default: 0). Can be a single number or comma-separated numbers (e.g., 7 or 0,1,2)')

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
                       help='Range of videos to process (e.g., "1-100" for first 100 videos)')
   
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
inf_result_csv_fp = inference_dir + f'/Task3+6_{model_name.split("/")[-1]}_{videos_range}.csv'

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
    # print("output_text:", output_text)
    # output_list = parse_json(output_text[0])
    output_list = raw_output[0]
    # print("output_list:", output_list)
    # output_text = '\"' + ', '.join(output_list) + '\"'
    output_text = '\"' + output_list + '\"'
    return output_text

# ================== Paths and file names ==================

import pandas as pd 
# def get_system_prompt():
#     system_prompt = '''
#         You are an assistant for analyzing seizure videos. 

#         You must use ONLY the following canonical feature names, each with its definition:

#         blank_stare: Does the patient exhibit a blank stare? 
#         arm_flexion: Does the patient flex their arms or arm at the elbows for at least a few video frames? 
#         arms_move_simultaneously: Do the patient's arms start moving approximately at the same time? 
#         occur_during_sleep: Is the patient sleeping at the beginning of the video?
#         close_eyes: Do the patient's eyes remain consistently closed or mostly closed throughout the video?
#         eye_blinking: Does the patient show rapid blinking of the eyes during the video?
#         tonic: The tonic phase is marked by a sudden onset of sustained stiffness or rigidity, usually lasting 5-20 seconds. This stiffness may be generalized, with all limbs held in fixed extension or flexion posture and can include stiffening of the head and axial body. It may also be focal involving a subset of body parts or just one body part at a time. Does this patient show tonic?
#         clonic: Clonic Phase: Rhythmic, jerking muscle contractions involving the limbs, face, and trunk. The jerking movements gradually slow before ceasing entirely. Clonic movements present as rhythmic, regular stereotyped contraction and relaxation of the affected body parts. Does this patient show clonic?
#         arm_straightening: Does the patient straighten or extend their arms or arm at the elbow for at least a few video frames?
#         figure4: Figure 4 refers to a tonic sustained posture where one arm is flexed while the other is extended at the same time. Does the patient exhibit a Figure 4 posture?
#         oral_automatisms: Does the patient exhibit repetitive, stereotyped mouth or tongue movements such as chewing, lip-smacking, or swallowing?
#         limb_automatisms: Does the patient exhibit repetitive, stereotyped limb movements such as fumbling, picking, rubbing or patting?
#         face_pulling: Does the patient exhibit unilateral sustained face-pulling movements?
#         face_twitching: Are there small muscle twitches observed on the patient's face?
#         head_turning: Does the patient forcibly or stiffly rotate their head to one side in the video?
#         asynchronous_movement: Do you observe the patient's limbs shake with variable frequency or amplitude with respect to one another?
#         pelvic_thrusting: Does the patient display repetitive, rhythmic, anteroposterior (forward-and-backward) movements of the hips?
#         full_body_shaking: Does the patient experience shaking of the entire body including arms, legs, torso?
    
#         Rules:
#         - Always use the canonical feature names exactly as written above.
#         - Never invent new features or paraphrase.
#         - The output format must be a JSON array of feature names in order of their first appearance.
#         - If no features are present, return [].
#         - Example: ["blank_stare", "tonic", "clonic"]
#     '''
        
#     return system_prompt

def get_task3_prompt():
    return '''
    Output the sequence of the any observed seizure symptoms of the patient in the video in chronological order. 
    The symptoms are limited to head_turning, blank_stare, close_eyes, eye_blinking, face_pulling, face_twitching, tonic, clonic, arm_straightening, arm_flexion, figure4, oral_automatisms, limb_automatisms, asynchronous_movement, pelvic_thrusting, full_body_shaking, arms_move_simultaneously. 
    If a symptom is not present in the video, it should not be included in the output.
    Example output: head_turning, arm_straightening, arm_flexion, tonic, clonic.
    Output only the seizure symptoms. Do not include any other text. 
    '''
    
def get_task6_prompt():
    return'''
    Generate a detailed semiological report for this seizure video, and the symptoms are limited to head_turning, blank_stare, close_eyes, eye_blinking, face_pulling, face_twitching, tonic, clonic, arm_straightening, arm_flexion, figure4, oral_automatisms, limb_automatisms, asynchronous_movement, pelvic_thrusting, full_body_shaking, arms_move_simultaneously, verbal_responsiveness, ictal_vocalization.
    For example: The patient is sleeping in bed. He lets out a loud groan and has versive head turn to the right. He has right upper extremity extension with left upper extremity flexion, followed by tonic-clonic activity. Later, the patient is unable to remember events preceding this seizure.
    Output the report in several sentences, plain language. Do not include other content. 
    '''

# ================== Utility functions ==================

def AskFreeText(video_clip_fp, log_file_fp):
    # task3 + task6 query
    if LOG:
        if not os.path.exists(log_file_fp):
            with open(log_file_fp, 'w') as f:
                f.write("video_fp,event_sequence\n")

    clip_seq_text = inference(model, video_clip_fp, get_task3_prompt())

    clip_report_text = inference(model, video_clip_fp, get_task6_prompt())

    if LOG:
        if not os.path.exists(log_file_fp):
            with open(log_file_fp, 'w') as f:
                f.write("video_fp,event_sequence,report\n")
        with open(log_file_fp, 'a') as f:
            f.write(f"{video_clip_fp},{clip_seq_text},{clip_report_text}\n")

    return clip_seq_text, clip_report_text

# ================== Main function ==================

def main():
    # List all files in the directory to check existence quickly
    input_clip_files = os.listdir(dataset_dir)
    
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
    if int(videos_range[1]) > len(input_clip_files):
        videos_range[1] = len(input_clip_files)
        # add warning
        print(f"Warning: videos_range[1] is greater than the number of videos, set to {len(input_clip_files)}")

    for video_clip_name in tqdm(input_clip_files[videos_range[0]-1 : videos_range[1]]):
        # skip if already in the CSV
        if not os.path.exists(inf_result_csv_fp):
            with open(inf_result_csv_fp, 'w') as f:
                f.write("video_name,event_sequence,report\n")
                
        with open(inf_result_csv_fp, 'r') as f:
            if video_clip_name in f.read():
                print(f"Video {video_clip_name} already processed. Skipping.")
                continue
        if LOG:
            log_dir = os.path.join(inference_dir, 'log')
            os.makedirs(log_dir, exist_ok=True)
            log_file_fp = os.path.join(log_dir, f"{video_clip_name}.csv")
        else:
            log_file_fp = None
        video_clip_fp = os.path.join(dataset_dir, video_clip_name)
        video_event_seq_list, clip_report_text = AskFreeText(video_clip_fp, log_file_fp)
        
        with open(inf_result_csv_fp, 'a') as f:
            f.write(f"{video_clip_name},{video_event_seq_list},{clip_report_text}\n")
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

