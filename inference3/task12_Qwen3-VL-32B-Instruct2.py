# import torch
# from PIL import Image
# from transformers import AutoModel, AutoTokenizer
# from torchvision.io import read_video    # pip install torchvision

import os
import json
import re
from tqdm import tqdm
import warnings

import argparse

# Suppress torchcodec warning
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*torchvision.*decoding.*deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.video_processing_utils")

default_model_cache_dir = os.path.join(os.path.dirname(__file__), 'model_cache')
default_output_dir = os.path.join(os.path.dirname(__file__), 'output')


def parse_arguments():
    # TODO:
    parser = argparse.ArgumentParser(description='Seizure Video Feature Extraction using Qwen3-VL-32B-Instruct')

    # GPU settings
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device ID(s) to use (default: 3). Can be a single number or comma-separated numbers (e.g., 7 or 0,1,2)')

    # TODO:
    # Model settings
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-VL-32B-Instruct',
                        help='Model name to use (default: Qwen/Qwen3-VL-32B-Instruct)')

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

    # # Logging settings
    # parser.add_argument('--disable_logs', type=lambda x: x.lower() in ('true', '1', 'yes'), default=True,
    #                    help='Disable individual log files for each video, keep only final result (default: True)')

    return parser.parse_args()


# Parse command line arguments
args = parse_arguments()

# Set GPU environment variable
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

# Set the directories/paths from arguments
################################################################################################
model_name = args.model_name
dataset_dir = args.dataset_dir
videos_range = args.videos_range

# inference files
inference_dir = args.output_dir
os.makedirs(inference_dir, exist_ok=True)
# if not args.disable_logs:
#     inference_log_dir = os.path.join(inference_dir, 'log')
#     os.makedirs(inference_log_dir, exist_ok=True)

# feature information

all_features = ['occur_during_sleep', 'blank_stare', 'close_eyes', 'eye_blinking',
                'tonic', 'clonic', 'arm_flexion', 'arm_straightening', 'figure4', 'oral_automatisms',
                'limb_automatisms',
                'face_pulling', 'face_twitching', 'head_turning', 'asynchronous_movement', 'pelvic_thrusting',
                'arms_move_simultaneously', 'full_body_shaking',
                # 'ictal_vocalization', 'verbal_responsiveness',
                ]

format_prompt_time = " Answer with 'yes' or 'no' and provide a justification for the answer.  Respond with exactly one JSON object in the format {\"answer\": \"yes\" or \"no\", \"justification\": \"brief explanation\", \"start_time\": \"MM:SS\" or \"N/A\"} and do not include any extra text outside of the JSON."
format_prompt_no_time = " Answer with 'yes' or 'no' and provide a justification for the answer. Respond with exactly one JSON object in the format {\"answer\": \"yes\" or \"no\", \"justification\": \"brief explanation\"} and do not include any extra text outside of the JSON."


# Function to clean and fix malformed JSON responses
def _strip_fence(s: str) -> str:
    s = s.strip()
    if "```" not in s:
        return s
    parts = s.split("```")
    # 取第一个代码块内容；若带语言标注（json/JSON），去掉那一行
    for i in range(1, len(parts), 2):
        block = parts[i].lstrip()
        lines = block.splitlines()
        if lines and lines[0].lower().strip() in {"json", "javascript", "js"}:
            block = "\n".join(lines[1:])
        return block
    return s


def _extract_first_balanced_braces(s: str) -> str | None:
    in_str = False
    esc = False
    depth = 0
    start = None
    for i, ch in enumerate(s):
        if ch == '"' and not esc:
            in_str = not in_str
        esc = (ch == "\\") and not esc
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return s[start:i + 1]
    return None


def clean_json_response(raw_response: str):
    if not raw_response:
        return None
    s = _strip_fence(raw_response)
    candidate = _extract_first_balanced_braces(s)
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # 兜底：从首个 { 起用 raw_decode 试一次
        dec = json.JSONDecoder()
        i = candidate.find("{")
        if i != -1:
            try:
                obj, _ = dec.raw_decode(candidate, i)
                return obj
            except json.JSONDecodeError:
                pass
        return None


# CSV file to read
inf_result_csv_fp = inference_dir + f'/Task1_{model_name.split("/")[-1]}_{videos_range}.csv'  # Output CSV (with extracted features)
# log_file = inference_log_dir + 'qwen_description_log.csv'      # Log file to record each prompt and answer


# Common video resolutions for target_size parameter:
# 1080p: (1920, 1080)
# 720p:  (1280, 720)
# 480p:  (854, 480)
# 360p:  (640, 360)
# 240p:  (426, 240)
################################################################################################
MAX_FRAMES = 120
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
# TODO:
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel

# TODO:
# Load base model first
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     model_name, torch_dtype="auto", device_map="auto"
# )
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    #attn_implementation="flash_attention_2",
    device_map="auto"
)
# Move model to GPU
# model = model.to('cuda')


# Load processor from the base model name, not the checkpoin
processor = AutoProcessor.from_pretrained(model_name, cache_dir=hf_cache_dir)

import os
import hashlib
import requests

# from IPython.display import Markdown, display
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
    # print(f"DEBUG: get_video_frames called with: {video_file_path}")
    # print(f"DEBUG: Video file exists: {os.path.exists(video_file_path)}")
    
    # os.makedirs(cache_dir, exist_ok=True)

    video_hash = hashlib.md5(video_file_path.encode('utf-8')).hexdigest()
    # print(f"DEBUG: Video hash: {video_hash}")

    frames_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_frames.npy')
    timestamps_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_timestamps.npy')
    # print(f"DEBUG: Cache files: {frames_cache_file}, {timestamps_cache_file}")

    if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
        # print(f"DEBUG: Loading from cache...")
        frames = np.load(frames_cache_file)
        timestamps = np.load(timestamps_cache_file)
        # print(f"DEBUG: Loaded frames shape: {frames.shape}, timestamps shape: {timestamps.shape}")
        return video_file_path, frames, timestamps

    # print(f"DEBUG: Reading video with torchvision...")
    try:
        # Read video using torchvision
        video_tensor, audio_tensor, video_info = read_video(video_file_path, pts_unit='sec')
        # print(f"DEBUG: Video tensor shape: {video_tensor.shape}")
        # print(f"DEBUG: Video info: {video_info}")
        
        total_frames = video_tensor.shape[0]
        fps = video_info['video_fps']
        # print(f"DEBUG: Total frames: {total_frames}, FPS: {fps}")

        # print("total_frames : ", total_frames)

        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
        # print(f"DEBUG: Frame indices: {indices[:10]}... (showing first 10)")

        # Extract selected frames
        selected_frames = video_tensor[indices]  # Shape: [num_frames, H, W, C]
        # print(f"DEBUG: Selected frames shape: {selected_frames.shape}")

        # Convert to numpy array and ensure uint8 format
        frames = selected_frames.numpy().astype(np.uint8)
        # print(f"DEBUG: Converted frames shape: {frames.shape}, dtype: {frames.dtype}")

        # Calculate timestamps for selected frames
        timestamps = np.array([idx / fps for idx in indices])
        # print(f"DEBUG: Timestamps shape: {timestamps.shape}")

        # print(f"DEBUG: Saving to cache...")
        np.save(frames_cache_file, frames)
        np.save(timestamps_cache_file, timestamps)
        # print(f"DEBUG: Cache saved successfully")

        return video_file_path, frames, timestamps
    except Exception as e:
        # print(f"DEBUG: Error in get_video_frames: {str(e)}")
        # print(f"DEBUG: Exception type: {type(e).__name__}")
        # print(f"DEBUG: Trying alternative video reading method...")
        
        # Try alternative method using OpenCV if torchvision fails
        try:
            import cv2
            # print(f"DEBUG: Attempting to read video with OpenCV...")
            cap = cv2.VideoCapture(video_file_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_file_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # print(f"DEBUG: OpenCV - Total frames: {total_frames}, FPS: {fps}")
            
            # Sample frames
            indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
            frames_list = []
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_list.append(frame_rgb)
                else:
                    # print(f"DEBUG: Warning - Could not read frame {idx}")
                    # Use the last successfully read frame
                    if frames_list:
                        frames_list.append(frames_list[-1])
                    else:
                        # Create a black frame as fallback
                        frames_list.append(np.zeros((480, 640, 3), dtype=np.uint8))
            
            cap.release()
            
            if not frames_list:
                raise ValueError("No frames could be read from video")
            
            frames = np.array(frames_list)
            timestamps = np.array([idx / fps for idx in indices])
            
            # print(f"DEBUG: OpenCV - Frames shape: {frames.shape}, timestamps shape: {timestamps.shape}")
            
            # Save to cache
            np.save(frames_cache_file, frames)
            np.save(timestamps_cache_file, timestamps)
            # print(f"DEBUG: OpenCV fallback successful")
            
            return video_file_path, frames, timestamps
            
        except Exception as cv_error:
            # print(f"DEBUG: OpenCV fallback also failed: {str(cv_error)}")
            traceback.print_exc()
            raise


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


# TODO:
def inference(video_path, prompt, max_new_tokens=None, max_pixels=602112, min_pixels=16 * 28 * 28):
    # print(f"DEBUG: inference called with video_path: {video_path}")
    # print(f"DEBUG: prompt length: {len(prompt)}")
    
    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS
    # print(f"DEBUG: max_new_tokens: {max_new_tokens}")
    
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
            {"type": "text", "text": prompt},
        ]}
    ]
    # print(f"DEBUG: Messages created: {len(messages)} message(s)")

    try:
        # print(f"DEBUG: Applying chat template...")
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        # print(f"DEBUG: Chat template applied successfully")
        # print(f"DEBUG: Input keys: {inputs.keys()}")
        # print(f"DEBUG: Input shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in inputs.items()]}")
        
        inputs = inputs.to(model.device)
        # print(f"DEBUG: Inputs moved to device: {model.device}")

        # Inference: Generation of the output
        # print(f"DEBUG: Starting model generation...")
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        # print(f"DEBUG: Generation completed, output shape: {generated_ids.shape}")
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        # print(f"DEBUG: Generated IDs trimmed, shape: {[ids.shape for ids in generated_ids_trimmed]}")
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # print(f"DEBUG: Output decoded, length: {len(output_text[0])}")
        # print(f"DEBUG: Full LLM output text:")
        print("=" * 80)
        print(output_text[0])
        print("=" * 80)
        
        return output_text[0]
    except Exception as e:
        # print(f"DEBUG: Error in inference: {str(e)}")
        # print(f"DEBUG: Exception type: {type(e).__name__}")
        traceback.print_exc()
        raise


# ================== Paths and file names ==================

import pandas as pd


def get_prompts():
    feature_names = all_features.copy()  # Copy the column names to feature_names

    # for feature in features_exluded:
    #     if feature in feature_names:
    #         feature_names.remove(feature)

    prompts = {}
    for feature in feature_names:
        if feature == 'blank_stare':
            # prompts[feature] = "Does the patient exhibit a blank stare (vacant or unfocused gaze) at any time during the event?  Do not include extra text in your output—only the answer."
            prompts[feature] = "Does the patient exhibit a blank stare? "
        if feature == 'arm_flexion':
            # prompts[feature] = "Does the patient maintain a sustained flexion of the arms at the elbows (i.e., holding them in a flexed position for a noticeable duration) during the seizure?  Do not include extra text in your output—only the answer."
            prompts[feature] = "Does the patient flex their arms or arm at the elbows for at least a few video frames? "
        # if feature == 'limb_movement_pattern':
        #     prompts[feature] = "Analyze the patient's upper limb movements in this video. Which of the following best describes the movements? 1.Thrashing/Flailing: Significant thrashing and flailing, somewhat asynchronous and variable. 2.Rhythmic Jerking: Stereotyped, rhythmic clonic jerking, potentially following a tonic phase. 3.Neither: The movements do not fit descriptions 1 or 2. Please answer with 1, 2, or 3. Do not include extra text in your output—only the answer."
        if feature == 'arms_move_simultaneously':
            # prompts[feature] = "Do the patient's hands start moving simultaneously? "
            prompts[feature] = "Do the patient's arms start moving approximately at the same time? "
        # if feature == 'gender':
        #     prompts[feature] = "Please identify the gender of the patient in the video. Please answer with \"female\" or \"male\". Do not include extra text in your output—only the answer."
        if feature == 'occur_during_sleep':
            # prompts[feature] = "Please determine if this seizure event occurs while the patient is asleep.  Do not include extra text in your output—only the answer."
            prompts[feature] = "Is the patient sleeping at the beginning of the video? "
        if feature == 'ictal_vocalization':
            # prompts[feature] = "Please check if the patient produces any vocalization (such as groaning, moaning, or screaming) during the event.  Do not include extra text in your output—only the answer."
            prompts[
                feature] = "Does the patient make any groaning, moaning, guttural sounds or do they utter stereotyped repetitive phrases? "
        if feature == 'close_eyes':
            # prompts[feature] = "Do the patient's eyes remain consistently closed or mostly closed throughout the event?  Do not include extra text in your output—only the answer."
            prompts[
                feature] = "Do the patient's eyes remain consistently closed or mostly closed throughout the video? "
        if feature == 'eye_blinking':
            # prompts[feature] = "Does the patient show repeated or rapid blinking of the eyes during the event?  Do not include extra text in your output—only the answer."
            prompts[feature] = "Does the patient show rapid blinking of the eyes during the video? "
        if feature == 'tonic':
            # prompts[feature] = "Please observe whether the patient has a prolonged, sustained muscle contraction (tonic phase).  Do not include extra text in your output—only the answer."
            prompts[
                feature] = "The tonic phase is marked by a sudden onset of sustained stiffness or rigidity, usually lasting 5–20 seconds. This stiffness may be generalized, with all limbs held in fixed extension or flexion posture and can include stiffening of the head and axial body. It may also be focal involving a subset of body parts or just one body part at a time. Does this patient show tonic? "
        if feature == 'clonic':
            # prompts[feature] = "Clonic movements typically involve repetitive, rhythmic jerking of muscles—marked by a contraction phase followed by relaxation, then repeating in a clear pattern. Please determine if the patient exhibits these repetitive, rhythmic jerking (clonic) movements, distinct from small or continuous trembling (tremor), at any point during the event.  Do not include extra text in your output—only the answer."
            prompts[
                feature] = "Clonic Phase: Rhythmic, jerking muscle contractions involving the limbs, face, and trunk. The jerking movements gradually slow before ceasing entirely. Clonic movements present as rhythmic, regular stereotyped contraction and relaxation of the affected body parts. Does this patient show clonic? "
        if feature == 'arm_straightening':
            # prompts[feature] = "Does the patient straighten or stiffen their arms (extended position)?  Do not include extra text in your output—only the answer."
            prompts[
                feature] = "Does the patient straighten or extend their arms or arm at the elbow for at least a few video frames? "
        if feature == 'figure4':
            # prompts[feature] = "Figure 4 refers to a specific posture or movement observed in a patient during a seizure, where one upper limb is extended (typically in a tonic stretch) while the other upper limb is flexed, forming a shape resembling the number 4. Please check if there is a 'figure 4' posture of the arms at any point.  Do not include extra text in your output—only the answer."
            prompts[
                feature] = "Figure 4 refers to a tonic sustained posture where one arm is flexed while the other is extended at the same time. Does the patient exhibit a Figure 4 posture? "
        if feature == 'oral_automatisms':
            # prompts[feature] = "Does the patient exhibit repetitive mouth or tongue movements such as chewing, lip-smacking, or swallowing?  Do not include extra text in your output—only the answer."
            prompts[
                feature] = "Does the patient exhibit repetitive,stereotyped mouth or tongue movements such as chewing, lip-smacking, or swallowing? "
        if feature == 'limb_automatisms':
            # prompts[feature] = "Are there repetitive, purposeless limb movements (e.g., fumbling, picking, patting, cycling) observed?  Do not include extra text in your output—only the answer."
            prompts[
                feature] = "Does the patient exhibit repetitive,stereotyped limb movements such as fumbling, picking, rubbing or patting?  "
        if feature == 'face_pulling':
            # prompts[feature] = "Does the patient's facial expression indicate grimacing or face-pulling movements?  Do not include extra text in your output—only the answer."
            prompts[feature] = "Does the patient exhibit unilateral sustained face-pulling movements? "
        if feature == 'face_twitching':
            # prompts[feature] = "Are there small, involuntary twitches or jerks observed on the patient's face?  Do not include extra text in your output—only the answer."
            prompts[feature] = "Are there small muscle twitches observed on the patient's face? "
        if feature == 'head_turning':
            # prompts[feature] = "Does the patient forcibly or stiffly rotate their head to one side during the event?   Do not include extra text in your output—only the answer."
            prompts[feature] = "Does the patient forcibly or stiffly rotate their head to one side in the video? "
        # if feature == 'tremor':
        #     # prompts[feature] = "Do you observe any rhythmic, trembling movement (tremor) in the patient's limbs or body?  Do not include extra text in your output—only the answer."
        #     prompts[feature] = "Do you observe any rhythmic, trembling movement (tremor) in the patient's limbs or body? "
        if feature == 'asynchronous_movement':
            prompts[
                feature] = "Do you observe the patient's limbs shake with variable frequency or amplitude with respect to one another? "
        if feature == 'pelvic_thrusting':
            # prompts[feature] = "Does the patient display any pelvic thrusting movements during the event?  Do not include extra text in your output—only the answer."
            prompts[
                feature] = "Does the patient display repetitive, rhythmic, anteroposterior (forward-and-backward) movements of the hips? "
        if feature == 'verbal_responsiveness':
            # prompts[feature] = "Please determine if the patient is able to respond verbally or demonstrate any comprehension during the event. Answer with 'yes' or 'no', if no one asks the patient any questions, answer 'NA.' Do not include extra text in your output—only the answer."
            prompts[
                feature] = "If the patient is addressed verbally by a different person, did they respond verbally in a coherent manner? Answer 'yes' or 'no'. If the patient is not addressed verbally by a different person, then the answer should be 'NA'."
        if feature == 'intensity_evolution':
            prompts[feature] = "Please determine if the intensity of the seizure changes over time. "
        # if feature == 'full_body_jerking':
        #     prompts[feature] = "Please determine if the patient exhibits full-body jerking movements during the event. "
        if feature == 'full_body_shaking':
            prompts[feature] = "Does the patient experience shaking of the entire body including arms, legs, torso? "
        # if feature == 'start_time':
        #     prompts[feature] = "At what time does the seizure start in the video? provide the timestamp in the format MM:SS (minutes and seconds). Do not include extra text in your output—only the timestamps."
        # if feature == 'end_time':
        #     prompts[feature] = "At what time does the seizure end in the video? provide the timestamp in the format MM:SS (minutes and seconds). Do not include extra text in your output—only the timestamps."

        # if feature in features_no_time:
        #     prompts[feature] = prompts[feature] + " " + format_prompt_no_time
        # # elif feature not in features_only_time:
        # #     prompts[feature] = prompts[feature] + " " + format_prompt_time
        # else:
        prompts[feature] = prompts[feature] + " " + format_prompt_no_time
    assert len(feature_names) == len(
        prompts), f"feature_names length {len(feature_names)} and prompt_list lengths {len(prompts)} does not match."
    return prompts


# ================== Utility functions ==================
def append_to_csv(csv_file, data, header=None):
    """
    Append the list 'data' to 'csv_file'. If 'csv_file' does not exist, write the 'header' row first.
    """
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if (not file_exists) and (header is not None):
            writer.writerow(header)
        writer.writerow(data)


def format_time(orig_time_str):
    """
    Extract MM:SS format from various time string formats.
    Handles formats like: "MM:SS", "MM:SS.xx", "MM:SS,xx", etc.
    Returns MM:SS format or "N/A" if invalid.
    """
    if not orig_time_str or orig_time_str.lower() == 'n/a':
        return "N/A"

    # Remove any extra whitespace
    orig_time_str = orig_time_str.strip()

    # Look for MM:SS pattern
    time_pattern = r'(\d{1,2}):(\d{2})'
    match = re.search(time_pattern, orig_time_str)

    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))

        # Validate ranges
        if 0 <= minutes <= 59 and 0 <= seconds <= 59:
            return f"{minutes:02d}:{seconds:02d}"

    return "N/A"


def ExtractFeatureByVLM(video_path, file_name, video_idx_info, log_csv, prompt_dict, json_error_log):
    """
    Extract features from the video by the VLM for each prompt in prompt_list,
    Return a list of extracted features in the same order as prompt_list.
    """

    # print(f"\n=== DEBUG: Starting feature extraction for {file_name} ===")
    print(f"Video path: {video_path}")
    print(f"Video exists: {os.path.exists(video_path)}")
    
    # extract feature for each prompt
    answer_dict = {}
    for feature in tqdm(prompt_dict.keys(), desc=f"Inferencing on video[{video_idx_info[0]}/{video_idx_info[1]}]",
                        total=len(prompt_dict.keys())):
        prompt = prompt_dict[feature]
        # print(f"\n--- DEBUG: Processing feature '{feature}' ---")
        print(f"Prompt: {prompt[:100]}...")

        answer_collected = False

        for retry_count in range(MAX_RETRIES):
            print(f"  Attempt {retry_count + 1}/{MAX_RETRIES}")
            raw_answer = None 
            try:
                print(f"  Getting video frames...")
                video_path, frames, timestamps = get_video_frames(video_path, num_frames=MAX_FRAMES)
                print(f"  Video frames shape: {frames.shape}")
                print(f"  Timestamps shape: {timestamps.shape}")

                print(f"  Running inference...")
                raw_answer = inference(video_path, prompt)
                print(f"  Raw answer length: {len(raw_answer) if raw_answer else 0}")
                print(f"  Raw answer preview: {raw_answer[:200] if raw_answer else 'None'}...")
                
                # Try direct JSON parsing first
                try:
                    print(f"  Attempting direct JSON parsing...")
                    answer_json = json.loads(raw_answer)
                    print(f"  Direct JSON parsing successful!")
                except json.JSONDecodeError as json_err:
                    print(f"  Direct JSON parsing failed: {json_err}")
                    # If direct parsing fails, try cleaning the response
                    print(f"  Attempting to clean response...")
                    answer_json = clean_json_response(raw_answer)
                    if answer_json is None:
                        # Log the cleaning failure too
                        json_error_log.write(f"Video: {file_name}\n")
                        json_error_log.write(f"Feature: {feature}\n")
                        json_error_log.write(f"Attempt: {retry_count + 1}\n")
                        json_error_log.write(f"Error: Failed to parse JSON even after cleaning\n")
                        json_error_log.write(f"Raw response: {raw_answer}\n")
                        json_error_log.write("-" * 50 + "\n\n")
                        json_error_log.flush()
                        raise ValueError("Failed to parse JSON even after cleaning")
                    print(f"  Cleaned JSON parsing successful!")

                print(f"  Parsed JSON: {answer_json}")
                answer = answer_json['answer']
                justification = answer_json['justification']
                print(f"  Extracted answer: {answer}")
                print(f"  Extracted justification: {justification[:100]}...")

                # Store all three values: answer, justification, start_time
                answer_dict[feature] = {
                    'answer': answer,
                    'justification': justification,
                }
                answer_collected = True
                print(f"  SUCCESS: Feature '{feature}' processed successfully!")
                break
            except Exception as e:
                print(f"  ERROR in attempt {retry_count + 1}: {str(e)}")
                print(f"  Raw VLM response: {raw_answer if raw_answer is not None else 'Not available'}")
                print(f"  Exception type: {type(e).__name__}")
                print(f"  Exception: {str(e)}. Retrying ({retry_count + 1}/{MAX_RETRIES})...")

                # Add more detailed error information for JSON parsing issues
                if isinstance(e, (json.JSONDecodeError, ValueError)) and raw_answer is not None:
                    print(f"  JSON parsing error details:")
                    print(f"    - Response length: {len(raw_answer) if raw_answer else 0}")
                    print(f"    - Response preview: {raw_answer[:200] if raw_answer else 'None'}...")
                    if hasattr(e, 'pos'):
                        print(f"    - Error position: {e.pos}")
                    if hasattr(e, 'lineno'):
                        print(f"    - Error line: {e.lineno}")

                traceback.print_exc()
                # time.sleep(10 * (retry_count + 1))

        if not answer_collected:
            print(f"  FAILED: All {MAX_RETRIES} attempts failed for feature '{feature}'")
            answer_dict[feature] = {
                'answer': "fail",
                'justification': "fail",
                # 'start_time': "fail"
            }
            # if not args.disable_logs:
            #     append_to_csv(
            #         log_csv,
            #         [file_name, prompt, "fail", "fail"]
            #     )
    
    # print(f"\n=== DEBUG: Feature extraction complete for {file_name} ===")
    print(f"Results: {answer_dict}")
    return answer_dict


# ================== Main function ==================

def main():
    prompt_dict = get_prompts()

    output_header = ['file_name']
    for feature in prompt_dict.keys():
        output_header.append(feature)
        output_header.append(f'justification_for_{feature}')

    # Create or append to JSON parsing error log file
    json_error_log_path = os.path.join(inference_dir, f"qwen3_vl_32b_task12_{args.gpu}.log")
    
    # Check if file exists and is not empty
    file_exists_and_has_content = os.path.exists(json_error_log_path) and os.path.getsize(json_error_log_path) > 0
    
    json_error_log = open(json_error_log_path, 'a', encoding='utf-8')
    
    # Only write header if file is new (doesn't exist or is empty)
    if not file_exists_and_has_content:
        json_error_log.write("JSON Parsing Error Log\n")
        json_error_log.write("=" * 50 + "\n\n")

    # List all files in the directory to check existence quickly
    input_videos_files = os.listdir(dataset_dir)

    # input_videos_files = ['S0005@3-18-2014@VA7610MM@nes_v6.mp4']
    input_videos_files = set(input_videos_files)
    input_videos_files = sorted(input_videos_files)

    # Check if result CSV exists and read processed files
    if os.path.exists(inf_result_csv_fp):
        result_df = pd.read_csv(inf_result_csv_fp)
        first_column_values = result_df.iloc[:, 0].tolist()
    else:
        first_column_values = []
        # Create result CSV with header only
        with open(inf_result_csv_fp, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(output_header)

    # make sure videos_range is valid
    videos_range = args.videos_range.split('-')
    videos_range = [int(videos_range[0]), int(videos_range[1])]
    if len(videos_range) != 2:
        raise ValueError("videos_range must be a comma-separated string of two numbers")
    if int(videos_range[0]) > int(videos_range[1]):
        raise ValueError("videos_range[0] must be less than videos_range[1]")
    if int(videos_range[0]) < 0:
        videos_range[0] = 1
        # add warning
        print(f"Warning: videos_range[0] is less than 0, set to 0")
    if int(videos_range[1]) > len(input_videos_files):
        videos_range[1] = len(input_videos_files)
        # add warning
        print(f"Warning: videos_range[1] is greater than the number of videos, set to {len(input_videos_files)}")
    video_list = input_videos_files[(videos_range[0] - 1): (videos_range[1])]

    for video_idx, file_name in enumerate(video_list):

        log_file = None
        # if not args.disable_logs:
        #     log_file = inference_log_dir + f'/{file_name}---log.csv'
        #     # Create log CSV with header if it doesn't exist
        #     # log_header = ["file_name", "prompt", "answer", "justification", "start_time"]
        #     log_header = ["file_name", "prompt", "answer", "justification"]
        #     with open(log_file, 'w', newline='', encoding='utf-8') as f:
        #         writer = csv.writer(f)
        #         writer.writerow(log_header)

        video_path = os.path.join(dataset_dir, file_name)
        row_to_write = [file_name]

        if file_name in first_column_values:
            print(file_name, "already processed")
            continue

        if file_name in input_videos_files:
            # If the file exists, process with Gemini
            # video_path = os.path.join(directory, file_name)
            print(f"Processing: {video_path}")
            # print(f"DEBUG: Video file exists: {os.path.exists(video_path)}")
            # print(f"DEBUG: Video file size: {os.path.getsize(video_path) if os.path.exists(video_path) else 'N/A'} bytes")
            try:
                answer_dict = ExtractFeatureByVLM(video_path, file_name, (video_idx + 1, len(video_list)), log_file,
                                                  prompt_dict, json_error_log)
                # print(f"DEBUG: ExtractFeatureByVLM returned: {answer_dict}")
                # Build row with proper structure: feature, justification, start_time for each feature
                for feature in prompt_dict.keys():
                    if feature in answer_dict:
                        # Extract the three values from the answer_dict
                        feature_data = answer_dict[feature]
                        row_to_write.append(feature_data['answer'])
                        # if feature in features_only_time:
                        #     continue
                        row_to_write.append(feature_data['justification'])
                        # if feature not in features_no_time:
                        #     row_to_write.append(feature_data['start_time'])
                    else:
                        # print(f"DEBUG: Feature '{feature}' not found in answer_dict")
                        row_to_write.extend(["fail", "fail"])
            except Exception as e:
                print(f"ERROR: Exception processing video {file_name}: {str(e)}")
                print(f"ERROR: Exception type: {type(e).__name__}")
                traceback.print_exc()
                # Create fail entries for all features (3 columns each: feature, justification, start_time)
                for _ in prompt_dict.keys():
                    row_to_write.extend(["fail", "fail"])
        else:
            # If the file does not exist, write empty features
            # Each feature needs 3 columns: feature, justification, start_time
            for _ in prompt_dict.keys():
                row_to_write.extend(["VideoNotExist", "VideoNotExist"])

        # Append to the output CSV (no header since it's already written)
        append_to_csv(inf_result_csv_fp, row_to_write)

    # Close the JSON error log file
    json_error_log.close()
    print(f"Processing is complete. Results are in '{inf_result_csv_fp}', JSON parsing errors logged in '{json_error_log_path}'.")


if __name__ == "__main__":
    print(f"Starting seizure video feature extraction...")
    print(f"GPU: {args.gpu}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Output: {inference_dir}")
    print(f"Videos range: {args.videos_range}")
    print(f"Max frames: {MAX_FRAMES}")
    print(f"FPS: {FPS}")
    # print(f"Log files: {'Disabled' if args.disable_logs else 'Enabled'}")
    print("-" * 50)

    main()

    # print("\n" + "="*50)
    # print("USAGE EXAMPLES:")
    # print("="*50)
    # print("# Use default settings:")
    # print("python ExtractFeature_qwen-2.5-vl-new.py")
    # print()
    # print("# Use different GPU:")
    # print("python ExtractFeature_qwen-2.5-vl-new.py --gpu 0")
    # print()
    # print("# Process more videos:")
    # print("python ExtractFeature_qwen-2.5-vl-new.py --max_videos 20")
    # print()
    # print("# Use different dataset:")
    # print("python ExtractFeature_qwen-2.5-vl-new.py --dataset_dir /path/to/videos")
    # print()
    # print("# Use different model:")
    # print("python ExtractFeature_qwen-2.5-vl-new.py --model_name Qwen/Qwen2.5-VL-14B-Instruct")
    # print()
    # print("# Enable individual log files for each video:")
    # print("python ExtractFeature_qwen-2.5-vl-new.py --disable_logs false")
    # print()
    # print("# See all options:")
    # print("python ExtractFeature_qwen-2.5-vl-new.py --help")
