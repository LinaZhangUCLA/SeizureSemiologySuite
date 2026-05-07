# import torch
# from PIL import Image
# from transformers import AutoModel, AutoTokenizer
# from torchvision.io import read_video    # pip install torchvision

import os
import json
from torch.utils.data import dataset
from tqdm import tqdm
import math

import argparse
import pandas as pd


report_dict = pd.read_csv("./../result/ground_truth/task6_report_annotation.csv", usecols=["file_name","report"], dtype=str, encoding="utf-8-sig")\
      .set_index("file_name")["report"].to_dict()
#print(report_dict)

LOG = True
# default_model_cache_dir = os.path.join(os.path.dirname(__file__), 'model_cache')
# default_output_dir = os.path.join(os.path.dirname(__file__), 'output')
default_model_cache_dir = '/mnt/SSD3/jiarui/model_cache'
default_output_dir = '/mnt/SSD3/jiarui/output'

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
    
   
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()
gpu_str = "".join(str(args.gpu).split(',')) 

# Set GPU environment variable
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Set the directories/paths from arguments
################################################################################################
model_name = args.model_name


# Set specific task directories

task3_HT_dataset_dir = os.path.join(args.dataset_dir, "task3_head_turning")
task3_AM_dataset_dir = os.path.join(args.dataset_dir, "task3_arm_movement")
task3_L_dataset_dir = os.path.join(args.dataset_dir, "task3_onset_body_part")
task4_dataset_dir = os.path.join(args.dataset_dir, "task4_feature_segments") 
task5_dataset_dir = os.path.join(args.dataset_dir, "task1256_segment_30s")
task6_dataset_dir = os.path.join(args.dataset_dir, "task1256_segment_30s")
task7_dataset_dir = os.path.join(args.dataset_dir, "task7_seizurevideos")

# Feature definitions dictionary
feature_definitions = {
    'blank_stare': "patient exhibit a blank stare ",
    'arm_flexion': "patient flex their arms or arm at the elbows for at least a few video frames ",
    'arms_move_simultaneously': "patient's arms start moving approximately at the same time ",
    'occur_during_sleep': "patient sleeping at the beginning of the video ",
    #'ictal_vocalization': "patient make any groaning, moaning, guttural sounds or utter stereotyped repetitive phrases ",
    'close_eyes': "patient's eyes remain consistently closed or mostly closed ",
    'eye_blinking': "patient shows rapid blinking of the eyes ",
    'tonic': "The tonic phase is marked by a sudden onset of sustained stiffness or rigidity, usually lasting 5–20 seconds. This stiffness may be generalized, with all limbs held in fixed extension or flexion posture and can include stiffening of the head and axial body. It may also be focal involving a subset of body parts or just one body part at a time. ",
    'clonic': "Clonic Phase: Rhythmic, jerking muscle contractions involving the limbs, face, and trunk. The jerking movements gradually slow before ceasing entirely. Clonic movements present as rhythmic, regular stereotyped contraction and relaxation of the affected body parts.",
    'arm_straightening': "patient straighten or extend their arms or arm at the elbow for at least a few video frames",
    'figure4': "Figure 4 refers to a tonic sustained posture where one arm is flexed while the other is extended at the same time.",
    'oral_automatisms': "patient exhibit repetitive, stereotyped mouth or tongue movements such as chewing, lip-smacking, or swallowing.",
    'limb_automatisms': "patient exhibit repetitive, stereotyped limb movements such as fumbling, picking, rubbing or patting.",
    'face_pulling': "patient exhibit unilateral sustained face-pulling movements.",
    'face_twitching': "small muscle twitches observed on the patient's face.",
    'head_turning': "patient forcibly or stiffly rotate their head to one side .",
    'asynchronous_movement': "patient's limbs shake with variable frequency or amplitude with respect to one another.",
    'pelvic_thrusting': "patient display repetitive, rhythmic, anteroposterior (forward-and-backward) movements of the hips.",
    #'verbal_responsiveness': "If the patient is addressed verbally by a different person, did they respond verbally in a coherent manner? Answer 'yes' or 'no'. If the patient is not addressed verbally by a different person, then the answer should be 'NA'.",
    'full_body_shaking': "patient experience shaking of the entire body including arms, legs, torso.",
    'start': "the seizure event starts in the video",
    'end': "the seizure event ends in the video"
}


videos_range = (args.videos_range).split('-')
task5_6_videos_range = (args.videos_range).split('-')
task3_HT_videos_range = '1-129'.split('-')
task3_AM_videos_range = '1-112'.split('-')
#task4_videos_range = (args.videos_range).split('-')

# inference files
inference_dir = args.output_dir
os.makedirs(inference_dir, exist_ok=True)

all_features = ['blank_stare','close_eyes','eye_blinking',
            'tonic','clonic','arm_flexion','arm_straightening','figure4','oral_automatisms','limb_automatisms',
            'face_pulling','face_twitching','head_turning','asynchronous_movement','pelvic_thrusting',
            'arms_move_simultaneously','full_body_shaking', 
            # 'ictal_vocalization', 'verbal_responsiveness','occur_during_sleep',
            ]

# CSV and log files for each task
task3_HT_result_csv_fp = os.path.join(inference_dir, f'Task3_HT_{model_name.split("/")[-1]}_{task3_HT_videos_range[0]}-{task3_HT_videos_range[1]}.csv')
task3_AM_result_csv_fp = os.path.join(inference_dir, f'Task3_AM_{model_name.split("/")[-1]}_{task3_AM_videos_range[0]}-{task3_AM_videos_range[1]}.csv')
task3_L_result_csv_fp = os.path.join(inference_dir, f'Task3_L_{model_name.split("/")[-1]}_{task3_AM_videos_range[0]}-{task3_AM_videos_range[1]}.csv')
task4_result_csv_fp = os.path.join(inference_dir, f'Task4_{model_name.split("/")[-1]}_{gpu_str}.csv')
task5_result_csv_fp = os.path.join(inference_dir, f'Task5_{model_name.split("/")[-1]}_{task5_6_videos_range[0]}-{task5_6_videos_range[1]}.csv')
task6_result_csv_fp = os.path.join(inference_dir, f'Task6_{model_name.split("/")[-1]}_{task5_6_videos_range[0]}-{task5_6_videos_range[1]}.csv')
task7_result_csv_fp = os.path.join(inference_dir, f'Task7_{model_name.split("/")[-1]}_{gpu_str}.csv')
# Log directories for each task
log_dir = os.path.join(inference_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

task3_log_dir = os.path.join(log_dir, 'task3')
task4_log_dir = os.path.join(log_dir, 'task4')
task5_log_dir = os.path.join(log_dir, 'task5')
task6_log_dir = os.path.join(log_dir, 'task6')
task7_log_dir = os.path.join(log_dir, 'task7')

for dir_path in [task3_log_dir, task4_log_dir, task5_log_dir, task6_log_dir]:
    os.makedirs(dir_path, exist_ok=True)


# Common video resolutions for target_size parameter:
# 1080p: (1920, 1080)
# 720p:  (1280, 720) 
# 480p:  (854, 480)
# 360p:  (640, 360)
# 240p:  (426, 240)
################################################################################################
MAX_FRAMES = 60
FPS = 2
TASK4_FPS = 1
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

    video_fps = FPS
    if "task4_feature_segments" in video_path:
        video_fps = TASK4_FPS

    messages = [
        {"role": "user", "content": [
            {
                "type": "video",
                "video": video_path,
                "max_pixels": max_pixels,
                "min_pixels": min_pixels,
                "total_pixels": max_pixels * MAX_FRAMES,
                "fps": video_fps,
            },
            {"type": "text", "text": query_prompt},
        ]}
    ]
    
    if "task7_seizurevideos" in video_path:
        messages = [
        {"role": "user", "content": [
            {
                "type": "video",
                "video": video_path,
                "nframes": MAX_FRAMES,  # <- 均匀抽取整段视频的 60 帧
                "max_pixels": max_pixels,
                "min_pixels": min_pixels,
                "total_pixels": max_pixels * MAX_FRAMES,
            },
            {"type": "text", "text": query_prompt},
        ]}
        ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    # print("video_kwargs:", video_kwargs, "messages:", messages)
    
    # print("video input:", video_inputs[0].shape)
    num_frames, _, resized_height, resized_width = video_inputs[0].shape
    # print("num of video tokens:", int(num_frames / 2 * resized_height / 28 * resized_width / 28))
    if "task7_seizurevideos" in video_path:
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, do_sample_frames=False, padding=True,return_tensors="pt").to('cuda')
    else:
        fps_inputs = video_kwargs['fps']
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt").to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    raw_output = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # print("raw_output:", raw_output[0])
    
    return raw_output[0]

# ================== Paths and file names ==================


def get_task5_prompt():
    return '''
    Output the sequence of the any observed seizure symptoms of the patient in the video in chronological order. 
    The symptoms are limited to head_turning, blank_stare, close_eyes, eye_blinking, face_pulling, face_twitching, tonic, clonic, arm_straightening, arm_flexion, figure4, oral_automatisms, limb_automatisms, asynchronous_movement, pelvic_thrusting, full_body_shaking, arms_move_simultaneously. 
    If a symptom is not present in the video, it should not be included in the output.
    Example output: head_turning, arm_straightening, arm_flexion, tonic, clonic.
    Output only the seizure symptoms. Do not include any other text. 
    '''

def get_task4_feature_prompt(feature: str):
    if feature == "start":
        return '''
        This video shows the start of a seizure event. Tell me the exact timestamp (MM:SS) when you first observe any seizure sign.
        Return only the JSON format: {"timestamp": "MM:SS"}
        '''
    elif feature == "end":
        return '''
        This video shows the end of a seizure event. Tell me the exact timestamp (MM:SS) when you observe the last seizure sign.
        Return only the JSON format: {"timestamp": "MM:SS"}
        '''
    else:
        description = feature_definitions.get(feature, "")
        return f'''
        This video shows when a patient {description}
        Tell me the exact timestamp (MM:SS) when this symptom first appears.
        Return only the JSON format: {{"timestamp": "MM:SS"}}
        '''

def get_task3_HT_prompt():
    return '''
    Does the patient's head turn to the patient's left or to the patient's right? 
    Answer with \"left\" or \"right\" only. Do not include any extra text. Return exactly one word: left or right.
    '''

def get_task3_AM_prompt():
    return '''
    Which arm of the patient is moving in the video? 
    Answer with \"left\" or \"right\" only. Do not include any extra text. Return exactly one word: left or right.
    '''

def get_task3_L_prompt():
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
    You are an expert clinician. Write a concise semiology description for this **seizure video**,
    focusing ONLY on observable **patient signs**.
    HARD RESTRICTIONS
    - Do NOT mention staff, restraints, bed/blanket/pillow, room devices, cameras, EEG leads/overlays, or timestamps.
    - Avoid vague words like “agitation”, “restlessness”, “discomfort”, or “adjusting position”.
    WHAT TO COVER (include an item ONLY if it is clearly visible in this video)
    • Early signs: blank stare, lip smacking, right/left head version, unilateral arm flexion/extension, tonic stiffening, clonic jerks.
    • Evolution & laterality: e.g., fencer posturing (left flexion with right extension), spread to bilateral tonic–clonic, automatisms, asynchronous shaking.
    STYLE
    - Write **1–3 short sentences in English only**, specific and minimal.
    - Examples (no labels): “Blank stare, then rightward head version with right arm extension; later bilateral tonic–clonic.”
        “Left arm flexion with right extension (fencer); rhythmic jerks follow. Unresponsive at the end.”
    Output ONLY the paragraph (no lists, no headers, no JSON).
    """

    
def get_task7_prompt_1(video_name):   
    return f"""
    Based on the patient’s seizure video and seizure semiology report, determine whether the patient has epileptic seizures (ES) or non-epileptic events (NES). Answer with 'ES' or 'NES’ and do not include any other text.
    seizrue report:
    {report_dict[video_name]}
    """

def get_task7_prompt_2():   
    return """
    Describe the patient's seizure symptoms in the video and diagnose whether it is an epileptic seizure (ES) or a non-epileptic event (NES).  Provide a description and answer with 'ES' or 'NES’. Respond with exactly one JSON object in the format {\"description\": \"...\",\"answer\": \"...\"} and do not include any extra text outside of the JSON."
    """    

# ================== Utility functions ==================

def query_task5_6(video_clip_fp, log_file_fp):
    # task5 + task6 query
    raw_output1 = inference(model, video_clip_fp, get_task5_prompt())
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

def normalize_direction_task3(ans: str) -> str:
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

def normalize_direction_task3_L(ans: str) -> str:
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

def query_task3(video_clip_fp, prompt):
    """
    Run a single left/right prompt and parse the answer (one word).
    Optionally log (file_name, prompt, answer) to per-video CSV.
    """
    raw_answer = inference(model, video_clip_fp, prompt)
    ans = normalize_direction_task3(raw_answer)
    return ans

def query_task3_L(video_clip_fp, prompt):
    raw_answer = inference(model, video_clip_fp, prompt)
    ans = normalize_direction_task3_L(raw_answer)
    return ans

def parse_json_task4(vlm_output):
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
        if isinstance(parsed, dict) and 'timestamp' in parsed:
            return parsed
        else:
            print(f"JSON parsed but missing required fields. Got: {parsed}")
            return {'timestamp': 'N/A'}
            
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Failed to parse JSON: {vlm_output}")
        print(f"Error: {e}")
        # Return default values if parsing fails
        return {'timestamp': 'N/A'}

def format_time_task4(raw_output):
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
    

def query_task4(video_clip_fp, feature):
    """Query the timing information for a specific feature in a video."""
    prompt = get_task4_feature_prompt(feature)
    raw_resp = inference(model, video_clip_fp, prompt)
    time_resp = parse_json_task4(raw_resp)
    
    start_time = format_time_task4(time_resp['start_time'])
    end_time = format_time_task4(time_resp['end_time'])
    
    return start_time, end_time, raw_resp


def get_duration_task4(start_time, end_time):
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


def parse_json_task7(vlm_output):
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
        if isinstance(parsed, dict) and 'answer' in parsed:
            return parsed["answer"]
        else:
            print(f"JSON parsed but missing required fields. Got: {parsed}")
            return {'N/A'}
            
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Failed to parse JSON: {vlm_output}")
        print(f"Error: {e}")
        # Return default values if parsing fails
        return {'N/A'}    

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


def init_csv(file_path, header):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write(header + "\n")

def load_processed_videos(csv_fp):
    if not os.path.exists(csv_fp):
        return set()
    with open(csv_fp, 'r') as f:
        return {line.split(',')[0] for line in f.readlines()[1:] if line.strip()}    

def main():
    global videos_range
    global task5_6_videos_range, task3_HT_videos_range, task3_AM_videos_range, task4_videos_range
    global task5_dataset_dir, task3_HT_dataset_dir, task3_AM_dataset_dir, task3_L_dataset_dir, task4_dataset_dir, task6_dataset_dir, task7_dataset_dir
    global task5_result_csv_fp, task3_HT_result_csv_fp, task3_AM_result_csv_fp, task4_result_csv_fp, task6_result_csv_fp
    
    task5_clip_fps = get_fp_list(task5_dataset_dir)
    task3_HT_clip_fps = get_fp_list(task3_HT_dataset_dir)
    task3_AM_clip_fps = get_fp_list(task3_AM_dataset_dir)
    task4_clip_fps = get_fp_list(task4_dataset_dir)
    task6_clip_fps = get_fp_list(task6_dataset_dir)
    task7_clip_fps = get_fp_list(task7_dataset_dir)

    
    # =============================================== task3=============================================================== #
    # try:
    #     # Get all task4 clips
    #     task3_HT_clip_fps = get_fp_list(task3_HT_dataset_dir)
    #     task3_AM_clip_fps = get_fp_list(task3_AM_dataset_dir)
    #     task3_L_clip_fps = get_fp_list(task3_L_dataset_dir)

    #     # Initialize task3 CSV files
    #     if not os.path.exists(task3_HT_result_csv_fp):
    #         with open(task3_HT_result_csv_fp, 'w') as f:
    #             f.write("video_name,head_turning_direction\n")
                
    #     if not os.path.exists(task3_AM_result_csv_fp):
    #         with open(task3_AM_result_csv_fp, 'w') as f:
    #             f.write("video_name,arm_movement_direction\n")
                
    #     if not os.path.exists(task3_L_result_csv_fp):
    #         with open(task3_L_result_csv_fp, 'w') as f:
    #             f.write("video_name,onset_body_part\n")

    #     # Process task4 videos
    #     if '7' in args.gpu:
    #         # Process head turning videos
    #         task3_HT_videos_range = validate_videos_range(task3_HT_clip_fps, task3_HT_videos_range)
    #         for video_clip_fp in tqdm(task3_HT_clip_fps[:], desc="Processing Task 3 Head Turning"):
    #             video_name = video_clip_fp.split('/')[-1]
    #             with open(task3_HT_result_csv_fp, 'r') as f:
    #                 if video_name in f.read():
    #                     print(f"Video {video_name} already processed for head turning. Skipping.")
    #                     continue
                
    #             try:
    #                 HT_ans = query_task3(video_clip_fp, get_task3_HT_prompt())
    #                 HT_ans = normalize_direction_task3(HT_ans)
    #                 with open(task3_HT_result_csv_fp, 'a') as f:
    #                     f.write(f"{video_name},{HT_ans}\n")
    #             except Exception as e:
    #                 print(f"Error processing video {video_name} for head turning: {e}")
    #                 with open(task3_HT_result_csv_fp, 'a') as f:
    #                     f.write(f"{video_name},N/A\n")
    #             #break
    #         # Process arm movement videos
    #         task3_AM_videos_range = validate_videos_range(task3_AM_clip_fps, task3_AM_videos_range)
    #         for video_clip_fp in tqdm(task3_AM_clip_fps[:], desc="Processing Task 3 Arm Movement"):
    #             video_name = video_clip_fp.split('/')[-1]
    #             with open(task3_AM_result_csv_fp, 'r') as f:
    #                 if video_name in f.read():
    #                     print(f"Video {video_name} already processed for arm movement. Skipping.")
    #                     continue
    #             try:
    #                 AM_ans = query_task3(video_clip_fp, get_task3_AM_prompt())
    #                 AM_ans = normalize_direction_task3(AM_ans)
    #                 with open(task3_AM_result_csv_fp, 'a') as f:
    #                     f.write(f"{video_name},{AM_ans}\n")
    #             except Exception as e:
    #                 print(f"Error processing video {video_name} for arm movement: {e}")
    #                 with open(task3_AM_result_csv_fp, 'a') as f:
    #                     f.write(f"{video_name},N/A\n")
    #             #break
    #         # Process onset body part videos
    #         task3_L_videos_range = validate_videos_range(task3_L_clip_fps, task3_AM_videos_range)
    #         for video_clip_fp in tqdm(task3_L_clip_fps[:], desc="Processing Task 3 Onset Body Part"):
    #             video_name = video_clip_fp.split('/')[-1]
    #             with open(task3_L_result_csv_fp, 'r') as f:
    #                 if video_name in f.read():
    #                     print(f"Video {video_name} already processed for onset body part. Skipping.")
    #                     continue
                
    #             try:
    #                 L_ans = query_task3_L(video_clip_fp, get_task3_L_prompt())
    #                 L_ans = normalize_direction_task3_L(L_ans)
    #                 with open(task3_L_result_csv_fp, 'a') as f:
    #                     f.write(f"{video_name},{L_ans}\n")
    #             except Exception as e:
    #                 print(f"Error processing video {video_name} for onset body part: {e}")
    #                 with open(task3_L_result_csv_fp, 'a') as f:
    #                     f.write(f"{video_name},N/A\n")
    #             #break
            
    # except Exception as e:
    #     print(f"Error in Task 3 processing: {e}")
    #     traceback.print_exc()
    # =============================================== task4 =============================================================== #
    # Initialize task4 CSV file
    # try:
    #     if not os.path.exists(task4_result_csv_fp):
    #         with open(task4_result_csv_fp, 'w') as f:
    #             f.write("video_name,feature,timestamp\n")

    #     # Process each feature folder
    #     #feature_folders = [d for d in os.listdir(task4_dataset_dir) if os.path.isdir(os.path.join(task4_dataset_dir, d))]
        
    #     feature_folders_split =  [
    #             ['blank_stare','close_eyes','eye_blinking',],
    #             ['tonic','clonic','arm_flexion',],
    #             ['arm_straightening','figure4','oral_automatisms',],
    #             ['limb_automatisms','face_pulling',],
    #             ['head_turning','asynchronous_movement',],
    #             ['pelvic_thrusting','arms_move_simultaneously',],
    #             ['full_body_shaking', 'start',],
    #             ['face_twitching','end'],
    #             # 'ictal_vocalization', 'verbal_responsiveness','occur_during_sleep',
    #             ]

    #     gpu_indices = [int(x) for x in str(args.gpu).split(',')]
    #     feature_folders = []
    #     for idx in gpu_indices:
    #         if idx < 0 or idx >= len(feature_folders_split):
    #             raise ValueError(f"Invalid GPU index {idx}, valid range: 0-{len(feature_folders_split)-1}")
    #         feature_folders.extend(feature_folders_split[idx])
    #     print(f"Selected features from GPUs {args.gpu}: {feature_folders}")

    #     processed = set()
    #     with open(task4_result_csv_fp, 'r') as f:
    #         try:
    #             next(f)  # 跳过表头
    #         except StopIteration:
    #             pass
    #         processed = set(','.join(line.strip().split(',')[:2]) for line in f)
        
        
    #     task4_log_file = os.path.join(task4_log_dir, f"task4_gpu{gpu_str}.log") 
    #     with open(task4_result_csv_fp, 'a') as csv_f, open(task4_log_file, 'a') as log_f:
    #         for feature in tqdm(feature_folders, desc="Processing features"):
    #             feature_path = os.path.join(task4_dataset_dir, feature)
    #             feature_log_path = os.path.join(task4_log_dir, f"{feature}.log")
                
    #             # Get all MP4 files in the feature folder

    #             video_files = [f for f in os.listdir(feature_path) if f.endswith('.mp4')]
    #             for video_name in tqdm(video_files, desc=f"Processing {feature} videos"):
    #                 video_path = os.path.join(feature_path, video_name)
    #                 key = f"{video_name},{feature}"
    #                 if key in processed:
    #                     print(f"Skipping already processed: {video_name} ({feature})")
    #                     continue           
                    
    #                 try:
    #                     # Use feature-specific prompt from definitions
    #                     prompt = get_task4_feature_prompt(feature)
    #                     raw_resp = inference(model, video_path, prompt)                       
    #                     log_f.write(f"Video: {video_name}\n")
    #                     log_f.write(f"Prompt Used: {prompt}\n")
    #                     log_f.write(f"Raw Response: {raw_resp}\n")
    #                     log_f.write("-" * 50 + "\n")

    #                     time_resp = parse_json_task4(raw_resp)
    #                     timestamp = format_time_task4(time_resp['timestamp'])
    #                     csv_f.write(f"{video_name},{feature},{timestamp}\n")
    #                     csv_f.flush()
    #                     processed.add(key)    
                                        
    #                     print(f"Successfully processed {video_name} for feature {feature}: timestamp={timestamp}")
                        
    #                 except Exception as e:
    #                     print(f"Error processing video {video_name} for feature {feature}: {e}")
    #                     csv_f.write(f"{video_name},{feature},N/A\n")
    #                     csv_f.flush()    
    #                     log_f.write(f"Error processing video {video_name} for feature {feature}: {e}\n")
    #                 #break
    #     print(f"Processing is complete. Results are in '{task4_result_csv_fp}'.")

    # except Exception as e:
    #     print(f"Error in Task 4 processing: {e}")
    #     traceback.print_exc()
    # =============================================== task5  =============================================================== #
    # try:

    #     def init_csv(file_path, header):
    #         if not os.path.exists(file_path):
    #             with open(file_path, 'w') as f:
    #                 f.write(header + "\n")

    #     init_csv(task5_result_csv_fp, "video_name,event_sequence")
    #     # Load already processed video names to avoid re-reading file each time
    #     def load_processed_videos(csv_fp):
    #         if not os.path.exists(csv_fp):
    #             return set()
    #         with open(csv_fp, 'r') as f:
    #             return {line.split(',')[0] for line in f.readlines()[1:] if line.strip()}

    #     task5_processed = load_processed_videos(task5_result_csv_fp)
    #     task5_videos_range = validate_videos_range(task5_clip_fps, videos_range)   
    #     with open(task5_result_csv_fp, 'a') as csv_f, open(os.path.join(task5_log_dir, "task5.log"), 'a') as log_f:
    #         for video_clip_fp in tqdm(task5_clip_fps[task5_videos_range[0]-1 : task5_videos_range[1]], desc="Processing Task 5"):
    #                 video_clip_name = video_clip_fp.split('/')[-1]
                            
    #                 if video_clip_name in task5_processed:
    #                     print(f"Video {video_clip_name} already processed for both tasks. Skipping.")
    #                     continue           
    #                 try: 
    #                         raw_output5 = inference(model, video_clip_fp, get_task5_prompt())
    #                         event_sequence = '\"' + raw_output5 + '\"'              
    #                         csv_f.write(f"{video_clip_name},{event_sequence}\n")
    #                         csv_f.flush() 
    #                         task5_processed.add(video_clip_name)                                       
    #                 except Exception as e:
    #                     print(f"Error processing video {video_clip_fp}: {e}")
    #                     csv_f.write(f"{video_clip_name},\"fail\"\n")
    #                     log_f.write(f"Error processing video {video_clip_name}: {e}\n")
    #                 #break
    #     print(f"Task 5 results are in: {task5_result_csv_fp}")  

    # except Exception as e:
    #     print(f"Error in Task 5 processing: {e}")
    #     traceback.print_exc()    

    # =============================================== task6 =============================================================== #
    # try:
    #     task5_videos_range = validate_videos_range(task5_clip_fps, videos_range)   #task6 uses the same video set as task5
    #     init_csv(task6_result_csv_fp, "video_name,report")
    #     task6_processed = load_processed_videos(task6_result_csv_fp)        
    #     os.makedirs(task6_log_dir, exist_ok=True)
    #     aggregate_log_fp = os.path.join(task6_log_dir, "task6.log")
    #     with open(task6_result_csv_fp, 'a', encoding='utf-8', newline='') as csv_f, open(aggregate_log_fp, 'a', encoding='utf-8') as log_f:    
    #         for video_clip_fp in tqdm(task5_clip_fps[task5_videos_range[0]-1 : task5_videos_range[1]], desc="Processing Task 6"):
    #             video_clip_name = video_clip_fp.split('/')[-1]
    #             if video_clip_name in task6_processed:
    #                 print(f"Video {video_clip_name} already processed for both tasks. Skipping.")
    #                 continue     
                    
    #             try:               
    #                     raw_output6 = inference(model, video_clip_fp, get_task6_prompt())
    #                     report = '\"' + raw_output6 + '\"'     
    #                     csv_f.write(f"{video_clip_name},{report}\n")
    #                     csv_f.flush()  
    #                     task6_processed.add(video_clip_name)                       
    #             except Exception as e:
    #                 print(f"Error processing video {video_clip_fp}: {e}")
    #                 log_f.write(f"Error processing video {video_clip_name}: {e}\n")
    #                 csv_f.write(f"{video_clip_name},\"fail\"\n")
    #             #break
    #     print(f"Task 6 results are in: {task6_result_csv_fp}")
    # except Exception as e:
    #     print(f"Error in Task 6 processing: {e}")
    #     traceback.print_exc()    



    # =============================================== task7 =============================================================== #
    def split_list_by_gpus(task7_clip_fps, gpu_indices):
        num_gpus_total = 8
        n = len(task7_clip_fps)
        chunk_size = math.ceil(n / num_gpus_total)
        chunks = []
        for i in range(num_gpus_total):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n)
            chunks.append(task7_clip_fps[start:end])
        selected = []
        for idx in gpu_indices:
            if 0 <= idx < num_gpus_total:
                selected.extend(chunks[idx])
            else:
                raise ValueError(f"GPU index {idx} out of range (0–7).")
        return selected
    
    try:

        init_csv(task7_result_csv_fp, "video_name,prediction_with_report,prediction_without_report")
        task7_processed = load_processed_videos(task7_result_csv_fp)        
        os.makedirs(task7_log_dir, exist_ok=True)
        aggregate_log_fp = os.path.join(task7_log_dir, f"task7_{gpu_str}.log")
        gpu_indices = [int(x) for x in str(args.gpu).split(',')]
        task7_clip_fps_selected = split_list_by_gpus(task7_clip_fps, gpu_indices)

        with open(task7_result_csv_fp, 'a', encoding='utf-8', newline='') as csv_f, open(aggregate_log_fp, 'a', encoding='utf-8') as log_f:    
            for video_clip_fp in tqdm(task7_clip_fps_selected, desc="Processing Task 7"):
                video_clip_name = video_clip_fp.split('/')[-1]
                if video_clip_name in task7_processed:
                    print(f"Video {video_clip_name} already processed for both tasks. Skipping.")
                    continue     
                    
                try:               
                        raw_output7 = inference(model, video_clip_fp, get_task7_prompt_1(video_clip_name))
                        prediction_with_report = '\"' + raw_output7 + '\"'     
                        raw_output7_2 = inference(model, video_clip_fp, get_task7_prompt_2())
                        prediction_without_report = '\"' + raw_output7_2 + '\"'  
                        log_f.write(f"{video_clip_name},\n,{prediction_with_report},\n,{prediction_without_report}\n")
                        log_f.flush()  

                        prediction_without_report =parse_json_task7(raw_output7_2)
                        csv_f.write(f"{video_clip_name},{raw_output7},{prediction_without_report}\n")
                        csv_f.flush()
                        task7_processed.add(video_clip_name)                       
                except Exception as e:
                    print(f"Error processing video {video_clip_fp}: {e}")
                    log_f.write(f"Error processing video {video_clip_name}: {e}\n")
                    csv_f.write(f"{video_clip_name},\"fail\",\"fail\"\n")
                #break
        print(f"Task 7 results are in: {task7_result_csv_fp}")
    except Exception as e:
        print(f"Error in Task 7 processing: {e}")
        traceback.print_exc()    
    

if __name__ == "__main__":
    print(f"Starting seizure video feature extraction...")
    print(f"GPU: {args.gpu}")
    print(f"Model: {args.model_name}")
    print(f"Output: {inference_dir}")
    print(f"Max frames: {MAX_FRAMES}")
    print(f"FPS: {FPS}")
    print("-" * 50)
    
    main()
