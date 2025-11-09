import pandas as pd
import os
from pathlib import Path

copy_from_path = "../metrics/middle_results"
copy_to_path = "../metrics_test/middle_results"

valid_patient_ids = pd.read_csv('../finetune/datasplit/fold0_val.csv')['patient_id'].astype(str).tolist()

# Convert to Path objects for easier handling
source_path = Path(copy_from_path)
target_path = Path(copy_to_path)

# Ensure target base directory exists
target_path.mkdir(parents=True, exist_ok=True)

# Walk through all directories and files in the source path
for root, dirs, files in os.walk(source_path):
    # Get relative path from source to maintain directory structure
    rel_path = Path(root).relative_to(source_path)
    target_dir = target_path / rel_path
    
    # Create corresponding directory in target
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each CSV file
    for file in files:
        if file.endswith('.csv'):
            source_file = Path(root) / file
            target_file = target_dir / file
            
            # Read CSV
            
            df = pd.read_csv(source_file)
            print(df.head(5))
            print(df.columns)
            # Check which column exists: file_name or video_name
            if 'file_name' in df.columns:
                column_name = 'file_name'
            elif 'video_name' in df.columns:
                column_name = 'video_name'
            else:
                print(f"Warning: {source_file} has neither 'file_name' nor 'video_name' column. Skipping.")
                continue
            
            # Filter rows where the column contains any valid_patient_id as substring
            mask = df[column_name].astype(str).apply(
                lambda x: any(patient_id in x for patient_id in valid_patient_ids)
            )
            filtered_df = df[mask].copy()
            
            # Save filtered CSV
            filtered_df.to_csv(target_file, index=False)
            print(f"Processed: {source_file} -> {target_file} ({len(filtered_df)}/{len(df)} rows kept)")

print("Done!")
