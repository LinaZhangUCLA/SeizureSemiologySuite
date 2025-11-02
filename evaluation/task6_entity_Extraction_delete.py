import pandas as pd
import os
import getpass
from openai import OpenAI
import time
from tqdm import tqdm

# --- OpenAI Client Configuration ---
try:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Please enter your OpenAI API key: ")
    client = OpenAI(api_key=api_key)
except Exception as e:
    print(f"Failed to initialize OpenAI client: {e}")
    client = None

# --- Semiology Features to be Evaluated ---
SEMIOLOGY_FEATURES = [
    'head_turning', 'tonic_posture', 'clonic_movements', 'limb_automatisms',
    'ictal_vocalization', 'asynchronous_movement', 'arm_flexion', 'arm_extension',
    'leg_bicycling', 'eye_closure', 'eye_blinking', 'face_pulling_grimacing',
    'face_twitching', 'unresponsive', 'post_ictal_confusion', 'dystonic_posturing',
    'tremulousness', 'staring', 'lip_smacking', 'pelvic_thrusting'
]

def create_prompt(report_text, feature_name):
    """Creates a prompt to ask the LLM to judge the presence of a single feature."""
    # Convert feature name to a more human-readable format, e.g., 'head_turning' -> 'Head Turning'
    formatted_feature = feature_name.replace('_', ' ').title()
    return f"""
    You are an expert clinical neurologist. Your task is to determine if a specific semiological feature is mentioned in the clinical report of a seizure episode.

    Analyze the following report and determine if it describes the feature: **"{formatted_feature}"**.

    **[Clinical Report]**
    "{report_text}"

    Does the report mention or describe "{formatted_feature}"?
    Your response must be a single word: ONLY "YES" or "NO".
    """

def judge_feature_with_openai(report_text, feature_name, model="gpt-4-turbo"):
    """
    Calls the OpenAI API to determine if a feature is present in the report.
    Returns 1 for "YES", 0 for "NO".
    """
    if not client or not isinstance(report_text, str) or not report_text.strip():
        return 0

    prompt = create_prompt(report_text, feature_name)

    for attempt in range(3): # Retry mechanism
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5
            )
            decision = response.choices[0].message.content.strip().upper()
            if "YES" in decision:
                return 1
            else:
                return 0
        except Exception as e:
            print(f"API call failed for feature '{feature_name}': {e}. Retrying attempt {attempt + 1}...")
            time.sleep(5)

    print(f"Error: API call failed after 3 retries for feature '{feature_name}'. Defaulting to 0.")
    return 0

# --- Main Program ---
if __name__ == "__main__":
    if not client:
        print("Execution halted: OpenAI client was not initialized successfully.")
        exit()

    inference_file = 'Task6_Qwen2.5-VL-7B-Instruct_all_merged_llmmerge.csv'
    output_file = 'inference_extracted_features_openai.csv'

    try:
        df_inference = pd.read_csv(inference_file)
    except FileNotFoundError:
        print(f"Error: The file {inference_file} was not found.")
        exit()

    all_features_data = []
    print(f"Starting feature extraction for {len(df_inference)} reports using OpenAI API...")

    for index, row in tqdm(df_inference.iterrows(), total=df_inference.shape[0], desc="Processing Reports"):
        file_name = row['file_name']
        report_text = row['report']

        extracted_features = {'file_name': file_name}
        for feature in SEMIOLOGY_FEATURES:
            # Call API for each feature and store the result (1 or 0)
            result = judge_feature_with_openai(report_text, feature)
            extracted_features[feature] = result
            time.sleep(0.5) # Add a small delay to avoid hitting API rate limits

        all_features_data.append(extracted_features)

    df_result = pd.DataFrame(all_features_data)
    df_result.to_csv(output_file, index=False)

    print(f"\nFeature extraction complete. Results saved to {output_file}")
    print("\nPreview of the generated file:")
    print(df_result.head())