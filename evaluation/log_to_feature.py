import os
import csv
import string
import pandas as pd
from itertools import zip_longest
from collections import defaultdict

# TODO - turn this into a YAML file
prompt_to_feature = {
    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Please identify the gender of the patient in the video. First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'female' or 'male'. Only use one of the possible answers: 'female' or 'male'. Do not include any extra text in your output—only the answer.""": "gender",
    
    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Please determine if this seizure event occurs while the patient is asleep. First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""": "occur_during_sleep",
    
    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Does the patient forcibly or stiffly rotate their head to one side during the event? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""": "head_turning",
    
    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Does the patient exhibit a blank stare (vacant or unfocused gaze)? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""": "blank_stare",
    
    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Do the patient's eyes remain consistently closed or mostly closed throughout the seizure? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""": "close_eyes",
    
    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Does the patient show repeated or rapid blinking of the eyes during the event? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""": "eye_blinking",
    
    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Does the patient's facial expression indicate grimacing or face-pulling movements? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""": "face_pulling",
    
    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Are there small, involuntary twitches or jerks observed on the patient's face? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""": "face_twitching",
    
    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Please observe whether the patient has sustained body stiffness (tonic phase). First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""": "tonic",
    
    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Clonic Phase: Rhythmic, jerking muscle contractions involving the limbs, face, and trunk. During this phase, individuals may bite their tongue or froth at the mouth. The episode typically lasts 30–90 seconds, though it can persist longer. The jerking movements gradually slow before ceasing entirely. Clonic movements present as rhythmic, alternating contraction-relaxation cycles (e.g., limb jerking), whereas tremors are characterized by continuous sinusoidal oscillations (e.g., hand shaking). If it is a tremor, it is not clonic. Does this patient show clonic? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""": "clonic",
    
    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Does the patient maintain a sustained flexion of the arms at the elbows (i.e., holding them in a flexed position for a noticeable duration) during the seizure? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""": "arm_flexion",
    
    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Does the patient straighten or stiffen their arms (extended position)? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""": "arm_straightening",
    
    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Figure 4 refers to a specific posture or movement observed in a patient during a seizure, where one upper limb is extended (typically in a tonic stretch) while the other upper limb is flexed, forming a shape resembling the number "4". Please check if there is a 'figure 4' posture of the arms at any point. First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""": "figure4",
    
    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Does the patient exhibit repetitive mouth or tongue movements such as chewing, lip-smacking, or swallowing? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""": "oral_automatisms",
    
    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Are there repetitive, purposeless limb movements (e.g., fumbling, picking, patting, cycling) observed? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""": "limb_automatisms",
    
    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Does the patient display any pelvic thrusting movements during the event? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""": "pelvic_thrusting",
    
    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Does the patient in the video experience jerking of the entire body, including the arms, legs, and torso? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""": "full_body_jerking",
    
    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Do you observe the patient lying in the bed shaking with variable intensity? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""": "tremor",
    
    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Does the patient show a continuous, intense, non-evolving motor pattern? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""": "intensity_pattern",
    
    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Analyze the patient's upper limb movements in this video. Which of the following best describes the movements? 1.Thrashing/Flailing: Significant thrashing and flailing, somewhat asynchronous and variable. 2.Rhythmic Jerking: Stereotyped, rhythmic clonic jerking, potentially following a tonic phase. 3.Neither: The movements do not fit descriptions 1 or 2. First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 1, 2, or 3. Only use one of the possible answers: 1, 2, or 3. Do not include any extra text in your output—only the answer.""": "limb_movements_pattern",
    
    """You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Do the patient's hands start moving simultaneously in the video? First, explain your answer in 3 sentences within parentheses i.e. ("answer"). Then write a new line in the form "\n". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.""": "hands_move_simultaneously"
}



##### Utils #####
def csv_to_pandas(filename):
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    df = pd.read_csv(filename)
    return df



def clean_answer(answer):
    acceptable_answers = {"yes", "no", "male", "female", "1", "2", "3"}

        
    # Remove punctuation and convert to lowercase
    answer = ''.join(c for c in answer if c not in string.punctuation).lower()
    
    print("in clean_answer:", answer)


    # Check if the answer is in the set of acceptable answers
    for acceptable_answer in acceptable_answers: # since it might have additional spaces and such
        print("*******")
        print(acceptable_answer, answer)
        if answer == 'fail':
            print("Failing answer, returning N/A")
            return "N/A"
        if acceptable_answer in answer:
            print(f"IT IS --> Answer: {acceptable_answer}")
        
            answer = acceptable_answer
            return answer

    return answer



def get_answer_from_output(output):
    ## if \n separator in output, then split the output into segments
    if '\n' in output:
        split_output = output.split('\n')
        answer = clean_answer(answer=split_output[-1])        
    else:        
        # remove punctuation and convert to lowercase
        answer = output[-10:].strip().lower()
        answer = clean_answer(answer)
            
    
    return answer


def get_answers_per_feature(video_segment_df):
    # Initialize a dictionary to hold the answers for each feature
    feature_answers = {feature: [] for feature in prompt_to_feature.values()}

    # Iterate through the rows of the DataFrame
    for _, row in video_segment_df.iterrows():
        prompt = row[1]
        output = row[2]
        
        
        # Get the corresponding feature name from the dictionary
        feature_name = prompt_to_feature.get(prompt, None)
        
        
        if feature_name:
            # get the answer from the output
            answer = get_answer_from_output(output)
            
            # Append the answer to the corresponding feature list
            if answer != "N/A":  # Only append valid answers
                feature_answers[feature_name].append(answer)
        else:
            raise ValueError(f"No feature mapping found for prompt: {prompt}")
        
    # now for each feature get the most common answer
    for feature, answers in feature_answers.items():
        if answers:
            # Use majority vote to determine the most common answer
            most_common_answer = majority_vote(pd.Series(answers))
            print(f"Most common answer for {feature}: {most_common_answer}")
            feature_answers[feature] = most_common_answer
        else:
            print("ANSWERS IS EMPTY", answers)
            feature_answers[feature] = "N/A"


    return feature_answers


def majority_vote(series):
    # Count the occurrence of each unique value
    counts = series.value_counts()
    # Return the value with the highest count; if there's a tie, the first one is returned
    return counts.idxmax() if not counts.empty else None


def remove_segment_suffix(video_segment_id):
    """
    Util function that removes the word "segment" from the video segment ID to match the full video ID.
    """
    
    # Define a function to remove the segment suffix from the video segment ID
    remove_segment_suffix = lambda filename: '@'.join(filename.split('@')[:-1] + [filename.split('@')[-1].split('_segment_')[0] + '.mp4']) if '_segment_' in filename else filename
    
    return remove_segment_suffix(video_segment_id)
        

def get_all_segment_feature_answers(df):
    """
    Returns a dictionary with video IDs as keys and a list of dictionaries as values.
    Each dictionary in the list contains feature answers for that video segment.    
    """
    
    final_answers_per_video = defaultdict(list)
    # for every video segment -> get all of the answers and clean them
    # Iterate through the unique video segment IDs in the first column
    for video_segment_id in df.iloc[:, 0].unique():
        print(f"Video Segment ID: {video_segment_id}")
        video_id = remove_segment_suffix(video_segment_id)
        
        print(f"Video ID: {video_id}")

        # Get the segment data for the current video segment ID
        segment_data = df[df.iloc[:, 0] == video_segment_id]
        
        
        # # For each video segment ID, get the answers for each feature
        # feature_answers = get_answers_per_feature(segment_data)
        
        # For each video segment ID, if the feature is yes then make the full video feature yes
        feature_answers = get_answers_per_feature(segment_data)
        # print(feature_answers)
        
        # Append the feature answers to the final answers dictionary
        final_answers_per_video[video_id].append(feature_answers)
        
    print("Final answers per video:", final_answers_per_video)

    
    return final_answers_per_video



def collapse_segment_answers_to_video(data):

    
    # Get all feature keys from the first dictionary
    feature_keys = data[0].keys()

    collapsed_dict = {}
    
    # Process each feature
    for feature in feature_keys:
        values = []
        
        # Collect all values for this feature
        for d in data:
            if feature in d and d[feature] is not None:
                values.append(d[feature])
        
        if not values:
            collapsed_dict[feature] = None
            continue
        
        # Rule 1: If any value is "yes", result is "yes"
        if "yes" in values:
            collapsed_dict[feature] = "yes"
        else:
            # Rule 2: Use majority voting
            # Count occurrences of each value
            value_counts = {}
            for val in values:
                value_counts[val] = value_counts.get(val, 0) + 1
            
            # Find the most common value
            max_count = max(value_counts.values())
            most_common = [val for val, count in value_counts.items() if count == max_count]
            
            
            # If there's a tie, take the first one alphabetically 
            # For 1,2 or 3 -> 1 or 2 will be chosen
            collapsed_dict[feature] = min(most_common)
    
    return collapsed_dict

   
   
   
def main():

    filename = 'SeizureSemiologyBench/internvl3_8B_segment_ido_log.csv'
    try:
        df = csv_to_pandas(filename)
        print("DataFrame loaded successfully:")
        print(df.head())
    except FileNotFoundError as e:
        print(e)
        
        
    
    
    # Get all segment feature answers i.e. {gender: "male", et}
    segment_answers_per_video = get_all_segment_feature_answers(df)
    
    # Open the output file once, outside the loop
    with open('output.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        header_written = False
        
        # Process each video
        for video_id, answers in segment_answers_per_video.items():
            
            # Get final features for this video
            final_features_per_video = collapse_segment_answers_to_video(answers)
            
            # Write header only once (for the first video)
            if not header_written:
                writer.writerow(['file_name'] + list(final_features_per_video.keys()))
                header_written = True
            
            # Write a single row for this video
            # Each dictionary value should be a single cell value, not unpacked
            writer.writerow([video_id] + list(final_features_per_video.values()))
            
            
   
if __name__ == "__main__":
    main()