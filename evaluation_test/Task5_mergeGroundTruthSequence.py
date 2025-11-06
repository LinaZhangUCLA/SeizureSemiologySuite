import pandas as pd
import re
import csv
from datetime import datetime

def parse_time_to_seconds(time_str):
    """Convert time string to seconds
    Args:
        time_str: Time string in format 'MM:SS:00'
    Returns:
        int: Time in seconds, or None if invalid
    """
    if not isinstance(time_str, str) or not time_str.strip():
        return None
    
    try:
        # Split the string into parts
        parts = time_str.strip().split(':')
        
        # Handle MM:SS:00 or M:SS format where MM/M is minutes, SS is seconds
        
        minutes = int(parts[0])
        seconds = int(parts[1])
        if minutes < 0 or seconds < 0 or minutes >= 60 or seconds >= 60:
            return None
        total_seconds = int(parts[0]) * 60 + int(parts[1])  # Convert MM:SS:00 to total seconds
            
        return total_seconds
            
        return None
    except (ValueError, IndexError):
        return None

def get_feature_sequence(row, feature_columns, exclude_features=None):
    """Get ordered sequence of features for a video
    Args:
        row: DataFrame row containing video annotations
        feature_columns: List of feature column names
        exclude_features: List of features to exclude (optional)
    Returns:
        tuple: (ordered_features, missing_times, invalid_times, has_any_features, all_features)
    """
    if exclude_features is None:
        exclude_features = []
    
    feature_times = []
    missing_times = []
    invalid_times = []
    has_any_features = False
    all_features = []  # Track all features marked as 'yes'
    
    for feature in feature_columns:
        # Skip excluded features
        if feature in exclude_features:
            continue
            
        # Check if feature is present (has 'yes' label)
        if str(row[feature]).strip().lower() == 'yes':
            has_any_features = True
            all_features.append(feature)
            time_col = f'start_time_for_{feature}'
            
            if time_col not in row:
                continue
                
            time_str = row[time_col]
            seconds = parse_time_to_seconds(str(time_str))
            
            if seconds is None:
                if pd.isna(time_str) or str(time_str).strip() == '':
                    missing_times.append(feature)
                else:
                    invalid_times.append(feature)
            else:
                feature_times.append((seconds, feature))
    
    # Sort features by start time
    feature_times.sort(key=lambda x: x[0])
    ordered_features = [f[1] for f in feature_times]
    
    return ordered_features, missing_times, invalid_times, has_any_features, all_features

def main():
    # Input and output paths
    input_path = 'result/ground_truth/task12_annotation.csv'
    output_path = 'result/ground_truth/task5_sequence_annotation.csv'
    output_path_vlm = 'result/ground_truth/task5_sequence_annotation_vlm.csv'
    missing_time_log = 'result/ground_truth/missingtime_error_logs.csv'
    invalid_time_log = 'result/ground_truth/invalid_time_logs.csv'
    no_features_path = 'result/ground_truth/task5_video_withoutfeatures.csv'
    debug_log = 'result/ground_truth/debug_log.txt'
    
    # Features to exclude from VLM output
    vlm_exclude_features = ['verbal_responsiveness', 'ictal_vocalization']
    
    # Read input CSV
    df = pd.read_csv(input_path, encoding='latin-1')
    
    # Filter out 'nan' entries
    df = df[df['file_name'].notna()]
    print(f"Total valid videos in input: {len(df)}")
    
    # Get feature columns (excluding metadata and justification columns)
    feature_columns = [col for col in df.columns 
                      if not any(x in col for x in ['file_name', 'justification', 'start_time', 'end_time'])]
    
    # Exclude "occur_during_sleep" since it has no start_time column
    feature_columns = [col for col in feature_columns if col != 'occur_during_sleep']
    
    # Initialize output data for both files
    output_data = []
    output_data_vlm = []
    missing_time_data = []
    invalid_time_data = []
    videos_without_features = []
    videos_without_features_vlm = []
    
    # Debug information
    # with open(debug_log, 'w') as f:
    #     f.write("Processing Details:\n")
    
    # Process each video for both output files
    for idx, row in df.iterrows():
        file_name = row['file_name']
        
        # Process for original output (all features)
        ordered_features, missing_times, invalid_times, has_any_features, all_features = get_feature_sequence(row, feature_columns)
        
        # Process for VLM output (excluding specified features)
        ordered_features_vlm, missing_times_vlm, invalid_times_vlm, has_any_features_vlm, all_features_vlm = get_feature_sequence(row, feature_columns, exclude_features=vlm_exclude_features)
        
        # Log processing details
        # with open(debug_log, 'a') as f:
        #     f.write(f"\nFile: {file_name}\n")
        #     f.write(f"Has any features: {has_any_features}\n")
        #     if has_any_features:
        #         f.write(f"All features: {all_features}\n")
        #         f.write(f"Ordered features: {ordered_features}\n")
        #         f.write(f"Missing times: {missing_times}\n")
        #         f.write(f"Invalid times: {invalid_times}\n")
        
        # Handle original output
        if not has_any_features:
            videos_without_features.append(idx)
        else:
            if ordered_features or all_features:
                output_data.append({
                    'file_name': file_name,
                    'event_sequence': '"' + (', '.join(ordered_features) if ordered_features else ', '.join(all_features)) + '"'
                })
            
            # Log missing times
            if missing_times:
                missing_time_data.append({
                    'file_name': file_name,
                    'missing_features': ', '.join(missing_times)
                })
            
            # Log invalid times
            if invalid_times:
                invalid_time_data.append({
                    'file_name': file_name,
                    'invalid_features': ', '.join(invalid_times)
                })
        
        # Handle VLM output
        if not has_any_features_vlm:
            videos_without_features_vlm.append(idx)
        else:
            if ordered_features_vlm or all_features_vlm:
                output_data_vlm.append({
                    'file_name': file_name,
                    'event_sequence': '"' + (', '.join(ordered_features_vlm) if ordered_features_vlm else ', '.join(all_features_vlm)) + '"'
                })
    
    # Print summary for original output
    print(f"\nSummary (Original):")
    print(f"Total valid videos processed: {len(df)}")
    print(f"Videos with sequences: {len(output_data)}")
    print(f"Videos without any features: {len(videos_without_features)}")
    print(f"Videos with missing times: {len(missing_time_data)}")
    print(f"Videos with invalid times: {len(invalid_time_data)}")
    
    # Print summary for VLM output
    print(f"\nSummary (VLM - excluding {', '.join(vlm_exclude_features)}):")
    print(f"Videos with sequences: {len(output_data_vlm)}")
    print(f"Videos without any features: {len(videos_without_features_vlm)}")
    
    # Verify total
    total_accounted = len(output_data) + len(videos_without_features)
    if total_accounted != len(df):
        print(f"\nWARNING: Not all videos accounted for!")
        print(f"Total videos: {len(df)}")
        print(f"Accounted for: {total_accounted}")
        
        # Find missing videos
        output_files = set([x['file_name'] for x in output_data])
        no_feature_files = set(df.iloc[videos_without_features]['file_name'])
        all_input_files = set(df['file_name'])
        missing_files = all_input_files - (output_files | no_feature_files)
        if missing_files:
            print("\nMissing files:")
            for f in missing_files:
                print(f"- {f}")
    
    # Save output files manually to avoid inconsistent quoting
    def write_csv_without_quotes(data, filepath):
        """Write CSV with no quotes on any values"""
        with open(filepath, 'w', encoding='utf-8') as f:
            # Write header
            f.write('file_name,event_sequence\n')
            # Write data rows
            for row in data:
                # Write row without quotes on either column
                f.write(f'{row["file_name"]},{row["event_sequence"]}\n')
    
    write_csv_without_quotes(output_data, output_path)
    print(f"\nSaved event sequences to {output_path}")
    
    write_csv_without_quotes(output_data_vlm, output_path_vlm)
    print(f"Saved VLM event sequences (excluding {', '.join(vlm_exclude_features)}) to {output_path_vlm}")
    
    # if missing_time_data:
    #     pd.DataFrame(missing_time_data).to_csv(missing_time_log, index=False)
    #     print(f"Saved missing time logs to {missing_time_log}")
    
    # if invalid_time_data:
    #     pd.DataFrame(invalid_time_data).to_csv(invalid_time_log, index=False)
    #     print(f"Saved invalid time logs to {invalid_time_log}")
    
    # if videos_without_features:
    #     # Save complete records for videos without features
    #     df_no_features = df.iloc[videos_without_features]
    #     df_no_features.to_csv(no_features_path, index=False)
    #     print(f"Saved {len(videos_without_features)} videos without features to {no_features_path}")
    
    # print(f"\nDetailed processing log saved to {debug_log}")

if __name__ == "__main__":
    main()