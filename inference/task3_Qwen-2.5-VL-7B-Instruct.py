# import torch
# from PIL import Image
# from transformers import AutoModel, AutoTokenizer
# from torchvision.io import read_video    # pip install torchvision

import os
import json
import re
from tqdm import tqdm

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Seizure Video Feature Extraction using Qwen2.5-VL')
    
    # GPU settings
    parser.add_argument('--gpu', type=str, default='3', 
                       help='GPU device ID(s) to use (default: 3)')
    
    # Model settings
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct',
                       help='Model name to use (default: Qwen/Qwen2.5-VL-7B-Instruct)')
    
    # Data settings
    parser.add_argument('--dataset_dir', type=str, 
                       default='/mnt/SSD3/tengyou/seizure_videos/segments/all_dataset/',
                       help='Directory containing seizure video files')
    parser.add_argument('--max_frames', type=int, default=60,
                       help='Maximum number of frames to extract from videos (default: 60)')
    parser.add_argument('--fps', type=int, default=2,
                       help='FPS for video processing (default: 2)')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='/mnt/SSD3/tengyou/inference/',
                       help='Output directory for results (default: /mnt/SSD3/tengyou/inference/)')
    parser.add_argument('--max_videos', type=int, default=10,
                       help='Maximum number of videos to process. Use -1 for all videos (default: 10)') 
    
    # Cache settings
    parser.add_argument('--cache_dir', type=str, default='/mnt/SSD3/tengyou/model_cache',
                       help='Directory for model cache (default: /mnt/SSD3/tengyou/model_cache)')
    
    # Processing settings
    parser.add_argument('--max_retries', type=int, default=10,
                       help='Maximum retries for failed prompts (default: 10)')
    parser.add_argument('--max_new_tokens', type=int, default=2048,
                       help='Maximum new tokens for generation (default: 2048)')
    
    return parser.parse_args()

# Parse command line arguments
args = parse_arguments()

# Set GPU environment variable
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Set the directories/paths from arguments
################################################################################################
root_dir = os.path.dirname(args.output_dir.rstrip('/')) + '/'
model_name = args.model_name
dataset_dir = args.dataset_dir


# inference files
inference_dir = args.output_dir
inference_result_dir = os.path.join(inference_dir, 'result')
os.makedirs(inference_result_dir, exist_ok=True)

# feature information

all_features = ['blank_stare','close_eyes','eye_blinking',
            'tonic','clonic','arm_flexion','arm_straightening','figure4','oral_automatisms','limb_automatisms',
            'face_pulling','face_twitching','head_turning','asynchronous_movement','pelvic_thrusting',
            'arms_move_simultaneously','ictal_vocalization', 'verbal_responsiveness', 'full_body_shaking',]

format_prompt_time = "Respond with exactly one JSON object in the format {\"answer\": \"yes\" or \"no\", \"start_time\": \"MM:SS\" or \"N/A\"} and do not include any extra text outside of the JSON."

# CSV file to read
inf_result_csv_fp = inference_dir + '/task3_result.csv' # Output CSV (with extracted features)

def clean_json_response(raw_response):
    """
    Clean malformed JSON responses from the VLM.
    Handles common issues like single quotes, extra text, etc.
    """
    if not raw_response:
        return None
    
    # Remove any leading/trailing whitespace
    raw_response = raw_response.strip()
    
    # Try to find JSON content within the response
    # Look for content between { and }
    start_idx = raw_response.find('{')
    end_idx = raw_response.rfind('}')
    
    if start_idx == -1 or end_idx == -1:
        print(f"No JSON brackets found in response: {raw_response}")
        return None
    
    # Extract the JSON part
    json_part = raw_response[start_idx:end_idx + 1]
    
    # Replace single quotes with double quotes
    json_part = json_part.replace("'", '"')
    
    # Fix common JSON formatting issues
    # Handle the case where the model might output "yes" or "no" literally
    json_part = json_part.replace('"yes" or "no"', '"yes" or "no"')
    
    # Remove any trailing commas before closing braces
    json_part = re.sub(r',(\s*})', r'\1', json_part)
    
    # Try to parse the cleaned JSON
    try:
        parsed = json.loads(json_part)
        print(f"Successfully cleaned and parsed JSON: {json_part}")
        return parsed
    except json.JSONDecodeError as e:
        print(f"JSON cleaning failed: {e}")
        print(f"Cleaned JSON part: {json_part}")
        
        # Try one more aggressive cleaning approach
        try:
            # Remove any non-ASCII characters that might cause issues
            json_part = ''.join(char for char in json_part if ord(char) < 128)
            # Try to fix common issues with quotes
            json_part = re.sub(r'([{,])\s*([^"]\w+)\s*:', r'\1 "\2":', json_part)
            parsed = json.loads(json_part)
            print(f"Successfully parsed after aggressive cleaning: {json_part}")
            return parsed
        except json.JSONDecodeError as e2:
            print(f"Even aggressive cleaning failed: {e2}")
            return None

################################################################################################
MAX_FRAMES = args.max_frames
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

# Set custom cache directory for model downloads
model_cache_dir = args.cache_dir
hf_cache_dir = os.path.join(model_cache_dir, 'huggingface')
modelscope_cache_dir = os.path.join(model_cache_dir, 'modelscope')
os.makedirs(hf_cache_dir, exist_ok=True)
os.makedirs(modelscope_cache_dir, exist_ok=True)

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
)
# Move model to GPU
model = model.to('cuda')

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


def get_video_frames(video_path, num_frames=128, cache_dir='.cache'):
    os.makedirs(cache_dir, exist_ok=True)

    # Handle file:// prefix
    if video_path.startswith('file://'):
        video_path = video_path[7:]  # Remove 'file://' prefix

    video_hash = hashlib.md5(video_path.encode('utf-8')).hexdigest()
    if video_path.startswith('http://') or video_path.startswith('https://'):
        video_file_path = os.path.join(cache_dir, f'{video_hash}.mp4')
        if not os.path.exists(video_file_path):
            download_video(video_path, video_file_path)
    else:
        video_file_path = video_path

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

    # np.save(frames_cache_file, frames)
    # np.save(timestamps_cache_file, timestamps)
    
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


def inference(video_path, prompt, max_new_tokens=None, max_pixels=602112, min_pixels=16 * 28 * 28):
    if max_new_tokens is None:
        max_new_tokens = args.max_new_tokens
    messages = [
        # {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {
                "type": "video",
                "video": video_path,
                "max_pixels": max_pixels,
                "min_pixels": min_pixels,
                "total_pixels": max_pixels * MAX_FRAMES,
                "fps": args.fps,
            },
            {"type": "text", "text": prompt},
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
    return output_text[0]

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
            # prompts[feature] = "Does the patient exhibit a blank stare (vacant or unfocused gaze) at any time during the event? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
            prompts[feature] = "Does the patient exhibit a blank stare? Answer with 'yes' or 'no'."
        if feature == 'arm_flexion':
            # prompts[feature] = "Does the patient maintain a sustained flexion of the arms at the elbows (i.e., holding them in a flexed position for a noticeable duration) during the seizure? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
            prompts[feature] = "Does the patient flex their arms or arm at the elbows for at least a few video frames? Answer with 'yes' or 'no'."
        # if feature == 'limb_movement_pattern':
        #     prompts[feature] = "Analyze the patient's upper limb movements in this video. Which of the following best describes the movements? 1.Thrashing/Flailing: Significant thrashing and flailing, somewhat asynchronous and variable. 2.Rhythmic Jerking: Stereotyped, rhythmic clonic jerking, potentially following a tonic phase. 3.Neither: The movements do not fit descriptions 1 or 2. Please answer with 1, 2, or 3. Do not include extra text in your output—only the answer."
        if feature == 'arms_move_simultaneously':
            # prompts[feature] = "Do the patient's hands start moving simultaneously? Answer with 'yes' or 'no'."
            prompts[feature] = "Do the patient's arms start moving approximately at the same time? Answer 'yes' or 'no'."
        # if feature == 'gender':
        #     prompts[feature] = "Please identify the gender of the patient in the video. Please answer with \"female\" or \"male\". Do not include extra text in your output—only the answer."
        if feature == 'occur_during_sleep':
            # prompts[feature] = "Please determine if this seizure event occurs while the patient is asleep. Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
            prompts[feature] = "Is the patient sleeping at the beginning of the video? Answer with 'yes' or 'no'."
        if feature == 'ictal_vocalization':
            # prompts[feature] = "Please check if the patient produces any vocalization (such as groaning, moaning, or screaming) during the event. Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
            prompts[feature] = "Does the patient make any groaning, moaning, guttural sounds or do they utter stereotyped repetitive phrases? Answer 'yes' or 'no'."
        if feature == 'close_eyes':
            # prompts[feature] = "Do the patient's eyes remain consistently closed or mostly closed throughout the event? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
            prompts[feature] = "Do the patient's eyes remain consistently closed or mostly closed throughout the video? Answer with 'yes' or 'no'."
        if feature == 'eye_blinking':
            # prompts[feature] = "Does the patient show repeated or rapid blinking of the eyes during the event? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
            prompts[feature] = "Does the patient show rapid blinking of the eyes during the video? Answer with 'yes' or 'no'."
        if feature == 'tonic':
            # prompts[feature] = "Please observe whether the patient has a prolonged, sustained muscle contraction (tonic phase). Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
            prompts[feature] = "The tonic phase is marked by a sudden onset of sustained stiffness or rigidity, usually lasting 5–20 seconds. This stiffness may be generalized, with all limbs held in fixed extension or flexion posture and can include stiffening of the head and axial body. It may also be focal involving a subset of body parts or just one body part at a time. Does this patient show tonic? Give an answer with 'yes' or 'no'."
        if feature == 'clonic':
            # prompts[feature] = "Clonic movements typically involve repetitive, rhythmic jerking of muscles—marked by a contraction phase followed by relaxation, then repeating in a clear pattern. Please determine if the patient exhibits these repetitive, rhythmic jerking (clonic) movements, distinct from small or continuous trembling (tremor), at any point during the event. Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
            prompts[feature] = "Clonic Phase: Rhythmic, jerking muscle contractions involving the limbs, face, and trunk. The jerking movements gradually slow before ceasing entirely. Clonic movements present as rhythmic, regular stereotyped contraction and relaxation of the affected body parts. Does this patient show clonic? Answer with 'yes' or 'no'."
        if feature == 'arm_straightening':
            # prompts[feature] = "Does the patient straighten or stiffen their arms (extended position)? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
            prompts[feature] = "Does the patient straighten or extend their arms or arm at the elbow for at least a few video frames? Answer with 'yes' or 'no'."
        if feature == 'figure4':
            # prompts[feature] = "Figure 4 refers to a specific posture or movement observed in a patient during a seizure, where one upper limb is extended (typically in a tonic stretch) while the other upper limb is flexed, forming a shape resembling the number 4. Please check if there is a 'figure 4' posture of the arms at any point. Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
            prompts[feature] = "Figure 4 refers to a tonic sustained posture where one arm is flexed while the other is extended at the same time. Does the patient exhibit a Figure 4 posture? Answer with 'yes' or 'no'."
        if feature == 'oral_automatisms':
            # prompts[feature] = "Does the patient exhibit repetitive mouth or tongue movements such as chewing, lip-smacking, or swallowing? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
            prompts[feature] = "Does the patient exhibit repetitive,stereotyped mouth or tongue movements such as chewing, lip-smacking, or swallowing? Answer with 'yes' or 'no'."
        if feature == 'limb_automatisms':
            # prompts[feature] = "Are there repetitive, purposeless limb movements (e.g., fumbling, picking, patting, cycling) observed? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
            prompts[feature] = "Does the patient exhibit repetitive,stereotyped limb movements such as fumbling, picking, rubbing or patting?  Answer with 'yes' or 'no'."
        if feature == 'face_pulling':
            # prompts[feature] = "Does the patient's facial expression indicate grimacing or face-pulling movements? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
            prompts[feature] = "Does the patient exhibit unilateral sustained face-pulling movements? Answer with 'yes' or 'no'."
        if feature == 'face_twitching':
            # prompts[feature] = "Are there small, involuntary twitches or jerks observed on the patient's face? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
            prompts[feature] = "Are there small muscle twitches observed on the patient's face? Answer with 'yes' or 'no'."
        if feature == 'head_turning':
            # prompts[feature] = "Does the patient forcibly or stiffly rotate their head to one side during the event?  Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
            prompts[feature] = "Does the patient forcibly or stiffly rotate their head to one side in the video? Answer with 'yes' or 'no'."
        # if feature == 'tremor':
        #     # prompts[feature] = "Do you observe any rhythmic, trembling movement (tremor) in the patient's limbs or body? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
        #     prompts[feature] = "Do you observe any rhythmic, trembling movement (tremor) in the patient's limbs or body? Answer with 'yes' or 'no'."
        if feature == 'asynchronous_movement':
            prompts[feature] = "Do you observe the patient's limbs shake with variable frequency or amplitude with respect to one another? Answer with 'yes' or 'no'."
        if feature == 'pelvic_thrusting':
            # prompts[feature] = "Does the patient display any pelvic thrusting movements during the event? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer."
            prompts[feature] = "Does the patient display repetitive, rhythmic, anteroposterior (forward-and-backward) movements of the hips? Answer 'yes' or 'no'."
        if feature == 'verbal_responsiveness':
            # prompts[feature] = "Please determine if the patient is able to respond verbally or demonstrate any comprehension during the event. Answer with 'yes' or 'no', if no one asks the patient any questions, answer 'NA.' Do not include extra text in your output—only the answer."
            prompts[feature] = "If the patient is addressed verbally by a different person, did they respond verbally in a coherent manner? Answer 'yes' or 'no'. If the patient is not addressed verbally by a different person, then the answer should be 'NA'."
        if feature == 'intensity_evolution':
            prompts[feature] = "Please determine if the intensity of the seizure changes over time. Answer with 'yes' or 'no'."
        # if feature == 'full_body_jerking':
        #     prompts[feature] = "Please determine if the patient exhibits full-body jerking movements during the event. Answer with 'yes' or 'no'."
        if feature == 'full_body_shaking':
            prompts[feature] = "Does the patient experience shaking of the entire body including arms, legs, torso? Answer with 'yes' or 'no'."
        # if feature == 'start_time':
        #     prompts[feature] = "At what time does the seizure start in the video? provide the timestamp in the format MM:SS (minutes and seconds). Do not include extra text in your output—only the timestamps."
        # if feature == 'end_time':
        #     prompts[feature] = "At what time does the seizure end in the video? provide the timestamp in the format MM:SS (minutes and seconds). Do not include extra text in your output—only the timestamps."
        
        # if feature in features_no_time:
        #     prompts[feature] = prompts[feature] + " " + format_prompt_no_time
        # # elif feature not in features_only_time:
        # #     prompts[feature] = prompts[feature] + " " + format_prompt_time
        # else:
        prompts[feature] = prompts[feature] + " " + format_prompt_time
    assert len(feature_names) == len(prompts), f"feature_names length {len(feature_names)} and prompt_list lengths {len(prompts)} does not match."
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


def ExtractFeatureByVLM(video_path, file_name, video_idx_info, log_csv, prompt_dict):
    """
    Extract features from the video by the VLM for each prompt in prompt_list,
    Return a list of extracted features in the same order as prompt_list.
    """

    
    # extract feature for each prompt
    answer_dict = {}
    for feature in tqdm(prompt_dict.keys(), desc=f"Inferencing on video[{video_idx_info[0]}/{video_idx_info[1]}]", total=len(prompt_dict.keys())):
        prompt = prompt_dict[feature]
        
        max_retries = args.max_retries
        answer_collected = False
        
        for retry_count in range(max_retries):
            try:
                video_path, frames, timestamps = get_video_frames(video_path, num_frames=MAX_FRAMES)
                raw_answer = inference(video_path, prompt)
                
                # Clean the JSON response before parsing
                cleaned_answer = clean_json_response(raw_answer)
                if cleaned_answer is None:
                    raise ValueError("Failed to clean JSON response")
                
                answer_json = json.loads(cleaned_answer)
                answer = answer_json['answer']
                start_time = format_time(answer_json['start_time'])
                append_to_csv(
                    log_csv,
                    [file_name, prompt, answer, start_time]
                )

                # Store all three values: answer, justification, start_time
                answer_dict[feature] = {
                    'answer': answer,
                    'start_time': start_time,
                }
                answer_collected = True
                break
            except Exception as e:
                print(f"Error in prompt for feature: {feature}: {prompt}")
                print(f"Raw VLM response: {raw_answer}")
                print(f"Cleaned response: {clean_json_response(raw_answer)}")
                print(f"Exception: {str(e)}. Retrying ({retry_count + 1}/{max_retries})...")
                
                # Add more detailed error information for JSON parsing issues
                if isinstance(e, (json.JSONDecodeError, ValueError)):
                    print(f"JSON parsing error details:")
                    print(f"  - Response length: {len(raw_answer) if raw_answer else 0}")
                    print(f"  - Response preview: {raw_answer[:200] if raw_answer else 'None'}...")
                    if hasattr(e, 'pos'):
                        print(f"  - Error position: {e.pos}")
                    if hasattr(e, 'lineno'):
                        print(f"  - Error line: {e.lineno}")
                
                traceback.print_exc()
                #time.sleep(10 * (retry_count + 1))
        
        if not answer_collected:
            answer_dict[feature] = {
                'answer': "fail",
                'start_time': "fail",
                # 'start_time': "fail"
            }
            append_to_csv(
                log_csv,
                [file_name, prompt, "fail", "fail"]
            )
    return answer_dict

def get_sequence_of_features(answer_dict, prompt_dict):
    feature_df = pd.DataFrame(columns=['feature', 'answer', 'start_time'])
    
    for feature in prompt_dict.keys():
        if feature in answer_dict:
            answer = answer_dict[feature]['answer']
            start_time = format_time(answer_dict[feature]['start_time'])
            feature_df.loc[len(feature_df)] = [feature, answer, start_time]
    
    # sort chronologically since format is strictly MM:SS
    feature_df = feature_df[feature_df['answer'] == 'yes'].sort_values(by='start_time')
    
    return feature_df['feature'].tolist()


# ================== Main function ==================

def main():
    prompt_dict = get_prompts()
    
    output_header = ['file_name', 'sequence_of_features']
    
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
    
    
    # Handle max_videos: -1 means all videos, otherwise limit to specified number
    if args.max_videos == -1:
        video_list = input_videos_files
    else:
        video_list = input_videos_files[:args.max_videos]
    for video_idx, file_name in enumerate(video_list):
        
        log_file = inference_result_dir + f'/{file_name}---log.csv'
        # Create log CSV with header if it doesn't exist
        # log_header = ["file_name", "prompt", "answer", "justification", "start_time"]
        log_header = ["file_name", "prompt", "answer", "start_time"]
        with open(log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(log_header)
            
        video_path = os.path.join(dataset_dir, file_name)
        row_to_write = [file_name]
        
        if file_name in first_column_values:
            print(file_name, "already processed")
            continue

        if file_name in input_videos_files:
            # If the file exists, process with Gemini
            #video_path = os.path.join(directory, file_name)
            print(f"Processing: {video_path}")
            try:
                answer_dict = ExtractFeatureByVLM(video_path, file_name, (video_idx + 1, len(video_list)), log_file, prompt_dict)
                # Build row with proper structure: feature, justification, start_time for each feature
                sequence_of_features = get_sequence_of_features(answer_dict, prompt_dict)
                row_to_write.append(sequence_of_features)
                
            except Exception as e:
                print(f"Error processing video {file_name}: {str(e)}")
                # Add empty sequence when there's an error
                row_to_write.append([])
        else:
            # If the file does not exist, write empty sequence
            row_to_write.append([])
        
        # Append to the output CSV (no header since it's already written)
        append_to_csv(inf_result_csv_fp, row_to_write)
    print(f"Processing is complete. Results are in '{inf_result_csv_fp}', logs in '{log_file}'.")

if __name__ == "__main__":
    print(f"Starting seizure video feature extraction...")
    print(f"GPU: {args.gpu}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Max videos: {args.max_videos}")
    print(f"Max frames: {args.max_frames}")
    print(f"FPS: {args.fps}")
    print("-" * 50)
    
    main()
    
    print("\n" + "="*50)
    print("USAGE EXAMPLES:")
    print("="*50)
    print("# Use default settings:")
    print("python ExtractFeature_qwen-2.5-vl-new.py")
    print()
    print("# Use different GPU:")
    print("python ExtractFeature_qwen-2.5-vl-new.py --gpu 0")
    print()
    print("# Process more videos:")
    print("python ExtractFeature_qwen-2.5-vl-new.py --max_videos 20")
    print()
    print("# Use different dataset:")
    print("python ExtractFeature_qwen-2.5-vl-new.py --dataset_dir /path/to/videos")
    print()
    print("# Use different model:")
    print("python ExtractFeature_qwen-2.5-vl-new.py --model_name Qwen/Qwen2.5-VL-14B-Instruct")
    print()
    print("# See all options:")
    print("python ExtractFeature_qwen-2.5-vl-new.py --help")
