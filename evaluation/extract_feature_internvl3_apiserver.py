import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu
import os
import time
import traceback
import pandas as pd
import csv
import numpy as np
from lmdeploy.serve.openai.api_client import APIClient
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy import pipeline, GenerationConfig
from decord import VideoReader, cpu
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.utils import encode_image_base64
from PIL import Image
import json
import re


# ================== Paths and file names ==================

experiment = "internvl3_8B_segments_ido_new_features"
directory = '/mnt/SSD3/tengyou/seizure_videos/segments/all_dataset/'  # Directory containing the videos
feature_file = experiment + '_feature.csv'  # Output CSV (with extracted features)
log_file = experiment + '_log.csv'  # Log file to record each prompt and answer


first_column_values = []
if os.path.exists(feature_file):
    df = pd.read_csv(feature_file)
    first_column_values = df.iloc[:, 0].tolist()

print(first_column_values)

api_client = APIClient(f'http://0.0.0.0:23333')
model_name = api_client.available_models[0]


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    # print("MAX_FRAME:", max_frame)
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    ## TEST
    # frame_indices = list(range(1, max_frame + 1, 2))
    # print("frame_indices:",frame_indices)

    return frame_indices


def load_video(video_path, bound=None, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    print("FPS:", fps)
    pixel_values_list, num_patches_list = [], []
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    imgs = []
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        imgs.append(img)
    return imgs


new_prompts = [
    "Does the patient lose awareness of self or surroundings at any point? Answer 'yes' or 'no'. Do not include extra text in your output -— only the answer.",
    "Does the patient appear dazed or confused during the event? Answer 'yes' or 'no'. Do not include extra text in your output -— only the answer.",
    "Does the patient interact with the environment inappropriately (e.g., purposeless grabbing)? Answer 'yes' or 'no'. Do not include extra text in your output -— only the answer.",
    "Is there forced eye opening or closure during the seizure? Answer 'yes' or 'no'. Do not include extra text in your output -— only the answer.",
    "Do you observe rapid, repetitive eyelid jerks (eyelid myoclonia)? Answer 'yes' or 'no'. Do not include extra text in your output -— only the answer.",
    "Are rhythmic horizontal or vertical eye oscillations (nystagmus) present? Answer 'yes' or 'no'. Do not include extra text in your output -— only the answer.",
    "Does the patient show hypersalivation or frothing? Answer 'yes' or 'no'. Do not include extra text in your output -— only the answer.",
    "Do brief, shock-like jerks of body parts occur? Answer 'yes' or 'no'. Do not include extra text in your output -— only the answer.",
    "Does the patient suddenly lose muscle tone and drop? Answer 'yes' or 'no'. Do not include extra text in your output -— only the answer.",
    "Does the patient exhibit violent, hypermotor movements soon after onset? Answer 'yes' or 'no'. Do not include extra text in your output -— only the answer.",
    "Is a 'fencing' posture (one arm extended, other flexed) seen? Answer 'yes' or 'no'. Do not include extra text in your output -— only the answer.",
    "Does the patient exhibit backward arching of the trunk (opisthotonus)? Answer 'yes' or 'no'. Do not include extra text in your output -— only the answer.",
    "Do brief (<2 s) flexor or extensor spasms of trunk/limbs occur? Answer 'yes' or 'no'. Do not include extra text in your output -— only the answer.",
    "Do you observe autonomic changes (flushing, pallor, apnea)? Answer 'yes' or 'no'. Do not include extra text in your output -— only the answer.",
    "Is there difficulty speaking or understanding language during or right after the seizure? Answer 'yes' or 'no'. Do not include extra text in your output -— only the answer.",
    "Is the patient confused or excessively sleepy after the event? Answer 'yes' or 'no'. Do not include extra text in your output -— only the answer.",
    "Does the patient develop temporary weakness or paralysis of a limb/body side after the seizure? Answer 'yes' or 'no'. Do not include extra text in your output -— only the answer."
]

new_features = [
    "impaired_awareness",
    "dazed_state",
    "inappropriate_interaction",
    "forced_eye_open_or_close",
    "eyelid_myoclonia",
    "nystagmus",
    "hypersalivation",
    "myoclonic_jerk",
    "atonic_drop",
    "hypermotor_activity",
    "fencing_posture",
    "opisthotonus",
    "epileptic_spasm",
    "autonomic_change",
    "language_disturbance",
    "postictal_confusion",
    "todds_paralysis"
]

feature_names = [
    "gender", "occur_during_sleep", "head_turning", "blank_stare", "close_eyes", "eye_blinking", "face_pulling", "face_twitching",
    "tonic", "clonic", "arm_flexion", "arm_straightening", "figure4", "oral_automatisms", "limb_automatisms",
    "pelvic_thrusting", "full_body_jerking", "tremor",
    "intensity_pattern", "limb_movements_pattern", "hands_move_simultaneously",
]

feature_names = ["forced_eye_open_or_close"]

prompt_list_orig = [
    """Please identify the gender of the patient in the video. Please answer with "female" or "male". Do not include extra text in your output—only the answer. """,
    """Please determine if this seizure event occurs while the patient is asleep. Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer.""",
    """Does the patient forcibly or stiffly rotate their head to one side during the event? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer.""",

    """Does the patient exhibit a blank stare (vacant or unfocused gaze)? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer. """,
    """Do the patient's eyes remain consistently closed or mostly closed throughout the seizure? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer. """,
    """Does the patient show repeated or rapid blinking of the eyes during the event? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer.""",
    """Does the patient's facial expression indicate grimacing or face-pulling movements? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer.""",
    """Are there small, involuntary twitches or jerks observed on the patient's face? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer.""",

    # """The tonic phase is characterized by sudden, sustained body stiffness, typically lasting 5 to 20 seconds. This rigidity can be generalized, causing limbs to become fixed in an extended or flexed posture and the trunk and back to straighten or arch, or it can be focal, limited to areas like the face (grimacing), eyes (deviation), or one side of the body. A temporary pause in breathing may occur, potentially causing to bluish skin or an initial cry.
    # Does this patient show tonic? Give an answer with 'yes' or 'no' and reason. Do not include extra text in your output—only the answer. """,

    """Please observe whether the patient has sustained body stiffness (tonic phase). Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer. """,
    """Clonic Phase: Rhythmic, jerking muscle contractions involving the limbs, face, and trunk. During this phase, individuals may bite their tongue or froth at the mouth. The episode typically lasts 30–90 seconds, though it can persist longer. The jerking movements gradually slow before ceasing entirely. Clonic movements present as rhythmic, alternating contraction-relaxation cycles (e.g., limb jerking), whereas tremors are characterized by continuous sinusoidal oscillations (e.g., hand shaking). If it is a tremor, it is not clonic. Dose this patient show clonic? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer.""",
    """Does the patient maintain a sustained flexion of the arms at the elbows (i.e., holding them in a flexed position for a noticeable duration) during the seizure? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer.""",
    """Does the patient straighten or stiffen their arms (extended position)? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer.""",
    """Figure 4 refers to a specific posture or movement observed in a patient during a seizure, where one upper limb is extended (typically in a tonic stretch) while the other upper limb is flexed, forming a shape resembling the number "4". Please check if there is a 'figure 4' posture of the arms at any point. Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer.""",
    """Does the patient exhibit repetitive mouth or tongue movements such as chewing, lip-smacking, or swallowing? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer.""",
    """Are there repetitive, purposeless limb movements (e.g., fumbling, picking, patting, cycling) observed? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer.""",

    """Does the patient display any pelvic thrusting movements during the event? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer. """,
    """Does the patient in the video experience jerking of the entire body, including the arms, legs, and torso? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer.""",
    """Do you observe the patient lying in the bed shaking with variable intensity? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer.""",
    # """Do you observe any rhythmic, trembling movement (tremor) in the patient's limbs or body? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer.""",

    """Does the patient show a continuous, intense, non-evolving motor pattern? Answer with 'yes' or 'no'. Do not include extra text in your output—only the answer.""",
    """Analyze the patient's upper limb movements in this video. Which of the following best describes the movements?

1.Thrashing/Flailing: Significant thrashing and flailing, somewhat asynchronous and variable.

2.Rhythmic Jerking: Stereotyped, rhythmic clonic jerking, potentially following a tonic phase.

3.Neither: The movements do not fit descriptions 1 or 2.

Please answer with 1, 2, or 3. Do not include extra text in your output—only the answer.""",
    """Do the patient's hands start moving simultaneously in the video? Answer with 'yes' or 'no'. Do not include extra text in your output"""
]

prompt_list = [
    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Please identify the gender of the patient in the video. First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'female' or 'male'. Only use one of the possible answers: 'female' or 'male'. Do not include any extra text in your output—only the answer.""",

    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Please determine if this seizure event occurs while the patient is asleep. First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""",

    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Does the patient forcibly or stiffly rotate their head to one side during the event? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""",

    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Does the patient exhibit a blank stare (vacant or unfocused gaze)? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""",

    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Do the patient's eyes remain consistently closed or mostly closed throughout the seizure? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""",

    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Does the patient show repeated or rapid blinking of the eyes during the event? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""",

    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Does the patient's facial expression indicate grimacing or face-pulling movements? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""",

    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Are there small, involuntary twitches or jerks observed on the patient's face? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""",

    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Please observe whether the patient has sustained body stiffness (tonic phase). First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""",

    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Clonic Phase: Rhythmic, jerking muscle contractions involving the limbs, face, and trunk. During this phase, individuals may bite their tongue or froth at the mouth. The episode typically lasts 30–90 seconds, though it can persist longer. The jerking movements gradually slow before ceasing entirely. Clonic movements present as rhythmic, alternating contraction-relaxation cycles (e.g., limb jerking), whereas tremors are characterized by continuous sinusoidal oscillations (e.g., hand shaking). If it is a tremor, it is not clonic. Does this patient show clonic? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""",

    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Does the patient maintain a sustained flexion of the arms at the elbows (i.e., holding them in a flexed position for a noticeable duration) during the seizure? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""",

    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Does the patient straighten or stiffen their arms (extended position)? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""",

    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Figure 4 refers to a specific posture or movement observed in a patient during a seizure, where one upper limb is extended (typically in a tonic stretch) while the other upper limb is flexed, forming a shape resembling the number "4". Please check if there is a 'figure 4' posture of the arms at any point. First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""",

    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Does the patient exhibit repetitive mouth or tongue movements such as chewing, lip-smacking, or swallowing? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""",

    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Are there repetitive, purposeless limb movements (e.g., fumbling, picking, patting, cycling) observed? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""",

    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Does the patient display any pelvic thrusting movements during the event? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""",

    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Does the patient in the video experience jerking of the entire body, including the arms, legs, and torso? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""",

    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Do you observe the patient lying in the bed shaking with variable intensity? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""",

    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Does the patient show a continuous, intense, non-evolving motor pattern? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""",

    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Analyze the patient's upper limb movements in this video. Which of the following best describes the movements? 1.Thrashing/Flailing: Significant thrashing and flailing, somewhat asynchronous and variable. 2.Rhythmic Jerking: Stereotyped, rhythmic clonic jerking, potentially following a tonic phase. 3.Neither: The movements do not fit descriptions 1 or 2. First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 1, 2, or 3. Only use one of the possible answers: 1, 2, or 3. Do not include any extra text in your output—only the answer.""",

    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Do the patient's hands start moving simultaneously in the video? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer."""
]

prompt_list = [
    """Looking at the patient's face, is the patient closing or squinting their eyes? Answer "yes" or "no". Do not include extra text in your output -— only the answer."""
]

new_prompts_tried = [
    """forced_eye_closure -- Are the patient's eyes closed? Answer "yes" or "no". Do not include extra text in your output -— only the answer.""",
    """impaired_awareness -- Does the patient lose awareness of themselves or their surroundings at any point? Answer "yes" or "no". Do not include extra text in your output -— only the answer. """
]

assert len(feature_names) == len(prompt_list), "feature_names and prompt_list lengths must match."


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


def extract_json_from_model_output(text):
    text = text.strip()
    text = re.sub(r'```json', '', text)
    text = re.sub(r'```$', '', text)
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    json_str = json_match.group(0)  # TODO: CAUSES AN ERROR
    return json_str


def ExtractFeatureByVLM(video_path, file_name, log_csv, start_time=None, end_time=None):
    """
    Extract features from the video by the VLM for each prompt in prompt_list,
    Return a list of extracted features in the same order as prompt_list.
    """
    imgs = load_video(video_path, num_segments=64)
    # Prepare a header for log CSV if needed
    log_header = ["file_name", "prompt", "answer"]
    # extract feature for each prompt
    features = []
    visual_content = []
    for img in imgs:
        visual_content.append({'type': 'image_url', 'image_url': {'max_dynamic_patch': 1, 'url': f'data:image/jpeg;base64,{encode_image_base64(img)}'}})

    for idx, prompt in enumerate(prompt_list):
        max_retries = 3
        answer_collected = False
        for retry_count in range(max_retries):
            try:
                question = ''
                for i in range(len(imgs)):
                    question = question + f'Frame{i+1}: {IMAGE_TOKEN}\n'

                question += prompt
                content = [{'type': 'text', 'text': question}]
                content.extend(visual_content)
                messages = [dict(role='user', content=content)]
                dict_gen_config = {}
                gen_config = GenerationConfig(top_p=0.8,
                                            top_k=1,
                                            temperature=0,
                                            max_new_tokens=1024)
                dict_gen_config['n'] = gen_config.n
                dict_gen_config['max_new_tokens'] = gen_config.max_new_tokens
                dict_gen_config['do_sample'] = gen_config.do_sample
                dict_gen_config['top_p'] = gen_config.top_p
                dict_gen_config['min_p'] = gen_config.min_p
                dict_gen_config['temperature'] = gen_config.temperature
                dict_gen_config['repetition_penalty'] = gen_config.repetition_penalty
                dict_gen_config['ignore_eos'] = gen_config.ignore_eos
                dict_gen_config['random_seed'] = gen_config.random_seed
                dict_gen_config['stop_words'] = gen_config.stop_words
                dict_gen_config['bad_words'] = gen_config.bad_words
                dict_gen_config['stop_token_ids'] = gen_config.stop_token_ids
                dict_gen_config['bad_token_ids'] = gen_config.bad_token_ids
                dict_gen_config['min_new_tokens'] = gen_config.min_new_tokens
                dict_gen_config['skip_special_tokens'] = gen_config.skip_special_tokens
                dict_gen_config['spaces_between_special_tokens'] = gen_config.spaces_between_special_tokens
                dict_gen_config['logprobs'] = gen_config.logprobs
                dict_gen_config['response_format'] = gen_config.response_format
                dict_gen_config['logits_processors'] = gen_config.logits_processors
                dict_gen_config['output_logits'] = gen_config.output_logits
                dict_gen_config['output_last_hidden_state'] = gen_config.output_last_hidden_state

                # GenerationConfig(n=1, max_new_tokens=1024, do_sample=False, top_p=0.8, top_k=1, min_p=0.0, temperature=0, repetition_penalty=1.0, ignore_eos=False, random_seed=None, stop_words=None, bad_words=None, stop_token_ids=None, bad_token_ids=None, min_new_tokens=None, skip_special_tokens=True, spaces_between_special_tokens=True, logprobs=None, response_format=None, logits_processors=None, output_logits=None, output_last_hidden_state=None)

                # potential_answers = api_client.chat_completions_v1(model=model_name,
                # messages=messages, gen_config=dict_gen_config)
                # print("*****potential_answers******", potential_answers)

                for item in api_client.chat_completions_v1(model=model_name,
                                                         messages=messages, gen_config=dict_gen_config):
                    answer = item['choices'][0]['message']['content']
                    output = "Q: " + prompt + "\n" + "A: " + answer + "\n"

                    print(output)
                    # Append this prompt/answer to log file
                    append_to_csv(
                        log_csv,
                        [file_name, prompt, answer],
                        header=log_header
                    )
                    # json_str = extract_json_from_model_output(answer)
                    # feat = json.loads(json_str).get("answer")
                    # features.append(feat)
                    # For eye feature -> reprompt
                    ### TODO will need a flag here
                    if "yes" in answer.lower(): # want to go around this so put random thigns instead of yes
                        print("EYES ARE CLOSED")
                        followup_prompt = ""
                        followup_prompt = f"Does the patient close their eyes for a long time and with force? Answer 'yes' or 'no'. Do not include extra text in your output -— only the answer."
                        content.extend([{'type': 'text', 'text': followup_prompt}])
                        messages = [dict(role='user', content=content)]

                        for item in api_client.chat_completions_v1(model=model_name,
                                                                 messages=messages, gen_config=dict_gen_config):
                            answer = item['choices'][0]['message']['content']
                            output = "Q: " + followup_prompt + "\n" + "A: " + answer + "\n"
                            print(output)
                            
                            # Append this follow-up prompt/answer to log file
                            append_to_csv(
                                log_csv,
                                [file_name, followup_prompt, answer],
                                header=log_header
                            )
                    answer_collected = True
                    break
            except Exception as e:
                print(f"Error in prompt: {prompt}")
                print(f"Exception: {str(e)}. Retrying ({retry_count + 1}/{max_retries})...")
                traceback.print_exc()
                # time.sleep(10 * (retry_count + 1))
        if not answer_collected:
            # If all retries failed, record "fail" for the feature
            features.append("fail")
            # Log as "fail"
            append_to_csv(
                log_csv,
                [file_name, prompt, "fail"],
                header=log_header
            )
    return features


def main():
    output_header = ["file_name"] + feature_names  # + ["label"]
    # List all files in the directory to check existence quickly
    dataset_files = os.listdir(directory)
    #####
    # files_to_review = [
    #     "E0008@1-22-2015@TA7641A3@sz_v1_1_segment_0.mp4",
    #     "E0008@1-22-2015@TA7641A3@sz_v1_1_segment_1.mp4",
    #     "E0008@1-22-2015@TA7641A3@sz_v1_1_segment_2.mp4",
    #     "E0008@1-22-2015@TA7641A3@sz_v1_1_segment_3.mp4"
    # ]
    
    files_to_review = [
        "L0001@5-31-2022@KA9601XM@sz_v1_1_segment_0.mp4",
        "L0001@5-31-2022@KA9601XM@sz_v1_1_segment_1.mp4",
        "L0001@5-31-2022@KA9601XM@sz_v1_1_segment_2.mp4",
        "L0001@5-31-2022@KA9601XM@sz_v1_1_segment_3.mp4",
        "L0001@5-31-2022@KA9601XM@sz_v1_1_segment_4.mp4",
        "L0001@5-31-2022@KA9601XM@sz_v1_1_segment_5.mp4",
        "L0001@5-31-2022@KA9601XM@sz_v1_1_segment_6.mp4"
    ]
   
    dataset_files = files_to_review
    # dataset_files = sorted(dataset_files)
    for file_name in dataset_files:
        match = re.search(r'segment_(\d+)\.mp4', file_name)
        segment_num = int(match.group(1))
        # TODO add back
        # if segment_num > 4:
        # print(file_name, " too long cut into 200s")
        # continue
        video_path = os.path.join(directory, file_name)
        print(f"Processing: {video_path}")
        row_to_write = []
        try:
            features = ExtractFeatureByVLM(video_path, file_name, log_file)
            row_to_write = [video_path] + features  # + [label]
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            fail_features = ["fail"] * len(feature_names)
            row_to_write = [video_path] + fail_features  # + [label]

        append_to_csv(feature_file, row_to_write, header=output_header)
    print(f"Processing is complete. Results are in '{feature_file}', logs in '{log_file}'.")


if __name__ == "__main__":
    main()