import pandas as pd
import glob
import os

# --- Configuration ---
INPUT_FOLDER_PATH = "metrics/Task2_feature_metrics" 
FILE_PATTERN = "*_feature_summary.csv"
OUTPUT_FILENAME = "metrics/task2_feature_metrics.csv"

# Define the desired feature order
FEATURE_ORDER = [
    'occur_during_sleep',
    'head_turning',
    'blank_stare',
    'close_eyes',
    'eye_blinking',
    'face_pulling',
    'face_twitching',
    'tonic',
    'clonic',
    'arm_straightening',
    'arm_flexion',
    'figure4',
    'oral_automatisms',
    'limb_automatisms',
    'asynchronous_movement',
    'pelvic_thrusting',
    'full_body_shaking',
    'arms_move_simultaneously',
    'verbal_responsiveness',
    'ictal_vocalization'
]

# Define model display names order
MODEL_NAMES = [
    'Qwen2.5-VL-7B',
    'InternVL3.5-8B',
    'Qwen2.5-VL-32B',
    'InternVL3.5-38B',
    'Qwen2.5-VL-72B',
    'Audio-flamingo-3',
    'Qwen2.5-Omni-7B',
    'Lingshu-32B',
    'Qwen3-VL-8B',
    'Qwen3-VL-32B'
]

# Mapping from display model names to actual file path names
MODEL_PATH_MAPPING = {
    'Qwen2.5-VL-7B': 'qwen2_5_vl_7B',
    'InternVL3.5-8B': 'internvl3_5_8B', 
    'Qwen2.5-VL-32B': 'qwen2_5_vl_32B',
    'InternVL3.5-38B': 'internvl3_5_38B',
    'Qwen2.5-VL-72B': 'qwen2_5_vl_72B',
    'Audio-flamingo-3': 'AF3',
    'Qwen2.5-Omni-7B': 'qwen2_5_omni_7B',
    'Lingshu-32B': 'lingshu_32B',
    'Qwen3-VL-8B': 'Qwen3-VL-8B',
    'Qwen3-VL-32B': 'Qwen3-VL-32B'
}

# Create reverse mapping (file path name to display name)
PATH_TO_DISPLAY_MAPPING = {v: k for k, v in MODEL_PATH_MAPPING.items()}
# ---------------------

def process_metrics_files(input_folder, file_pattern, output_filename):
    """
    Reads multiple CSV files from a folder, extracts metrics for each feature, 
    and creates a wide-format table with model as rows and feature_metric as columns.
    """
    
    # 1. Find all matching files
    search_path = os.path.join(input_folder, file_pattern)
    print(f"Searching for files at: {search_path}")
    
    all_files = glob.glob(search_path)
    
    print(f"Found {len(all_files)} files:")
    for f in all_files:
        print(f"  - {f}")
    
    if not all_files:
        print(f"\nNo files found matching the pattern: {search_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Does the folder exist? {os.path.exists(input_folder)}")
        if os.path.exists(input_folder):
            print(f"Files in folder: {os.listdir(input_folder)}")
        return

    # 2. Dictionary to store data for each model (using display name as key)
    all_model_data = {}

    # 3. Process each file
    for file_path in all_files:
        try:
            # Extract model name from filename
            filename = os.path.basename(file_path)
            # Remove '_feature_summary.csv' suffix to get file model name
            file_model_name = filename.replace("_feature_summary.csv", "")
            
            # Map to display name
            display_model_name = PATH_TO_DISPLAY_MAPPING.get(file_model_name, file_model_name)
            
            print(f"\nProcessing: {filename}")
            print(f"File model name: {file_model_name}")
            print(f"Display model name: {display_model_name}")

            # Read the CSV file
            df = pd.read_csv(file_path)
            
            print(f"Columns in CSV: {df.columns.tolist()}")
            print(f"Number of rows: {len(df)}")
            print(f"First few rows:\n{df.head()}")
            
            # Remove 'AVG' row if it exists (we want individual features only)
            original_len = len(df)
            df = df[df['feature'] != 'AVG']
            print(f"Rows after removing AVG: {len(df)} (removed {original_len - len(df)})")
            
            # Create a dictionary to store this model's data
            model_data = {'model': display_model_name}
            
            # For each feature (row), extract the three metrics
            for idx, row in df.iterrows():
                feature_name = row['feature']
                
                # Extract and round the three metrics to 2 decimal places
                rouge = round(row['rouge1_f1_mean'], 2)
                bleu = round(row['bleu_corpus'], 2)
                bertscore = round(row['berts_f1_mean'], 2)
                
                # Create column names: feature_metric (using 'rouge' instead of 'rouge1_f1_mean')
                model_data[f"{feature_name}_rouge"] = rouge
                model_data[f"{feature_name}_bleu"] = bleu
                model_data[f"{feature_name}_bertscore"] = bertscore
                
                print(f"  Feature: {feature_name} - ROUGE: {rouge}, BLEU: {bleu}, BERTScore: {bertscore}")
            
            all_model_data[display_model_name] = model_data
            print(f"Successfully processed {display_model_name}")
            
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
        except KeyError as e:
            print(f"Error: Missing column {e} in file {file_path}. Check CSV format.")
            if 'df' in locals():
                print(f"Available columns: {df.columns.tolist()}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {file_path}: {e}")
            import traceback
            traceback.print_exc()

    # 4. Create the final DataFrame and save it
    if all_model_data:
        # Convert to list of dictionaries in the desired model order
        ordered_model_data = []
        for model_name in MODEL_NAMES:
            if model_name in all_model_data:
                ordered_model_data.append(all_model_data[model_name])
            else:
                print(f"Warning: Model '{model_name}' not found in processed data")
        
        final_df = pd.DataFrame(ordered_model_data)
        
        print(f"\nFinal DataFrame shape: {final_df.shape}")
        
        # Reorder columns according to FEATURE_ORDER
        # Create ordered column list: model, then features in order with their 3 metrics each
        ordered_cols = ['model']
        for feature in FEATURE_ORDER:
            # Add the three metrics for this feature in order
            ordered_cols.append(f"{feature}_rouge")
            ordered_cols.append(f"{feature}_bleu")
            ordered_cols.append(f"{feature}_bertscore")
        
        # Filter to only include columns that actually exist in the dataframe
        ordered_cols = [col for col in ordered_cols if col in final_df.columns]
        
        # Reorder the dataframe
        final_df = final_df[ordered_cols]
        
        print(f"Ordered columns (first 10): {final_df.columns.tolist()[:10]}")
        print(f"Model order: {final_df['model'].tolist()}")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        final_df.to_csv(output_filename, index=False)
        print(f"\n✓ Successfully processed {len(ordered_model_data)} files.")
        print(f"✓ Output saved to {output_filename}")
        print(f"✓ Output columns: {len(final_df.columns)} columns")
        print(f"✓ Features found: {(len(final_df.columns) - 1) // 3}")  # Subtract 'model' column
    else:
        print("\n✗ No data was successfully processed to save to a file.")

# Run the function
process_metrics_files(INPUT_FOLDER_PATH, FILE_PATTERN, OUTPUT_FILENAME)