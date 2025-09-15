import pandas as pd
import os
import getpass
from openai import OpenAI
import time
from tqdm import tqdm

# --- OpenAI Client Setup ---
try:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Please enter your OpenAI API key: ")
    client = OpenAI(api_key=api_key)
except Exception as e:
    print(f"Failed to initialize OpenAI client: {e}")
    client = None

def create_prompt(reference_report, inference_report):
    """Creates the prompt for evaluating report similarity."""
    return f"""
    You are an expert clinical neurologist evaluating an AI-generated report of a patient's seizure episode. Your task is to assess the semantic and clinical similarity of the AI's report compared to a ground truth report written by a human expert.

    Please provide a single floating-point score from 0.0 to 1.0 based on the following criteria:
    - 1.0 (Excellent): The AI-generated report accurately captures all critical semiological features mentioned in the ground truth. It is factually consistent and contains no hallucinations.
    - 0.7-0.9 (Good): The AI-generated report captures the majority of the key semiological features but may miss minor details or use less precise clinical terminology.
    - 0.4-0.6 (Moderate): The AI-generated report identifies some correct semiological features but misses significant ones or contains minor factual inaccuracies.
    - 0.1-0.3 (Poor): The AI-generated report only describes general, non-specific events and fails to identify most of the crucial clinical signs mentioned in the ground truth.
    - 0.0 (Failure): The AI-generated report is completely irrelevant, factually incorrect, or hallucinates information not supported by the ground truth.

    Evaluate the following two reports:

    **[Ground Truth Report]**
    "{reference_report}"

    **[AI-Generated Report]**
    "{inference_report}"

    Based on your evaluation, provide a single floating-point number representing the similarity score. Your response must contain ONLY the numerical score and nothing else.
    """

def get_similarity_score(reference, inference, model="gpt-4-turbo"):
    """Calls the OpenAI API to get the similarity score."""
    if not client:
        return None

    if not isinstance(reference, str) or not isinstance(inference, str) or not reference or not inference:
        return 0.0

    prompt = create_prompt(reference, inference)
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            score_text = response.choices[0].message.content.strip()
            return float(score_text)
        except ValueError:
            print(f"Warning: Could not convert API response '{score_text}' to a float. Retrying...")
            time.sleep(2)
        except Exception as e:
            print(f"API call failed: {e}. Retrying attempt {attempt + 1}...")
            time.sleep(5)
    print("Error: API call failed after 3 retries.")
    return None

if __name__ == "__main__":
    if not client:
        print("Exiting: OpenAI client could not be initialized.")
        exit()

    # --- 1. Load and Merge Data ---
    reference_file = 'task6_annotation.csv'
    inference_file = 'Task6_Qwen2.5-VL-7B-Instruct_all_merged_llmmerge.csv'
    output_file = 'similarity_scores.csv'

    try:
        df_ref = pd.read_csv(reference_file)
        df_inf = pd.read_csv(inference_file)

        df_ref.rename(columns={'report': 'reference_report'}, inplace=True)
        df_inf.rename(columns={'report': 'inference_report'}, inplace=True)

        df_merged = pd.merge(df_ref, df_inf, on='file_name', how='inner')
    except FileNotFoundError as e:
        print(f"Error: File not found at {e.filename}. Please ensure the file exists in the correct path.")
        exit()

    # --- 2. Iterate and Get Scores ---
    scores = []
    print(f"Starting similarity score evaluation for {len(df_merged)} records...")

    for index, row in tqdm(df_merged.iterrows(), total=df_merged.shape[0]):
        score = get_similarity_score(row['reference_report'], row['inference_report'])
        scores.append(score)
        time.sleep(1)

    # --- 3. Save Results ---
    df_merged['similarity_score'] = scores
    df_merged.to_csv(output_file, index=False)

    print(f"\nSimilarity evaluation complete. Results saved to {output_file}")
    print("\nPreview of the generated file:")
    print(df_merged[['file_name', 'similarity_score']].head())