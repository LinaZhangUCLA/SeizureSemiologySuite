import pandas as pd
import re
from collections import defaultdict
import sys
import os

def time_to_seconds(time_str):
    """Convert MM:SS format to total seconds"""
    if pd.isna(time_str) or time_str == '':
        return 0
    minutes, seconds = map(int, str(time_str).split(':'))
    return minutes * 60 + seconds

def extract_base_filename(filename):
    """Extract base filename by removing _segment_X suffix"""
    # Remove _segment_X.mp4 pattern
    pattern = r'_segment_\d+\.mp4$'
    base_name = re.sub(pattern, '.mp4', filename)
    return base_name

def merge_segments(input_csv_path, output_csv_path):
    """
    Merge video segments and calculate new start/end times
    
    Formula: end_time = last_segment_num * 25 + end_time_of_last_segment
    """
    # Read the CSV file
    df = pd.read_csv(input_csv_path)
    
    # Filter out rows with missing file_name
    df = df.dropna(subset=['file_name'])
    
    # Group segments by base filename
    segments_by_video = defaultdict(list)
    
    for _, row in df.iterrows():
        base_filename = extract_base_filename(row['file_name'])
        segments_by_video[base_filename].append(row)
    
    # Process each video group
    merged_results = []
    
    for base_filename, segments in segments_by_video.items():
        # Sort segments by segment number
        segments.sort(key=lambda x: int(re.search(r'_segment_(\d+)', x['file_name']).group(1)))
        
        # Get start time from first segment (segment_0) in seconds
        start_time_seconds = time_to_seconds(segments[0]['start_time'])
        
        # Get the last segment number and its end time
        last_segment = segments[-1]
        last_segment_num = int(re.search(r'_segment_(\d+)', last_segment['file_name']).group(1))
        last_segment_end_time = last_segment['end_time']
        
        # Calculate merged end time: (last_segment_num * 25) + end_time_of_last_segment
        last_end_seconds = time_to_seconds(last_segment_end_time)
        total_end_seconds = (last_segment_num * 25) + last_end_seconds
        
        merged_results.append({
            'file_name': base_filename,
            'start_seconds': start_time_seconds,
            'end_seconds': total_end_seconds
        })
    
    # Create output DataFrame and save
    output_df = pd.DataFrame(merged_results)
    output_df.to_csv(output_csv_path, index=False)
    print(f"Merged segments saved to: {output_csv_path}")
    print(f"Processed {len(merged_results)} videos from {len(df)} segments")
    
def main():
    # Change to SeizureSemiologyBench directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    # For each of these VLM inferences
    model_names = [
        "InternVL3_5-8B",
        "InternVL3_5-38B", 
        "Lingshu-32B",
        "Qwen2.5-Omni-7B",
        "Qwen2.5-VL-7B-Instruct",
        "Qwen2.5-VL-32B-Instruct",
        "Qwen2.5-VL-72B-Instruct"
    ]
    
    # Merge the times for each model
    for model_name in model_names:
        input_path = f"result/vlm_inference/{model_name}/Task5_{model_name}_all.csv"
        print(input_path)
        output_path = f"result/vlm_inference/{model_name}/Task5_{model_name}_all_merged.csv"
        merge_segments(input_path, output_path)

if __name__ == "__main__":
    main()

### Output: ###
# ../result/vlm_inference/InternVL3_5-8B/Task5_InternVL3_5-8B_all.csv
# Merged segments saved to: ../result/vlm_inference/InternVL3_5-8B/Task5_InternVL3_5-8B_all_merged.csv
# Processed 294 videos from 1625 segments
# ../result/vlm_inference/InternVL3_5-38B/Task5_InternVL3_5-38B_all.csv
# Merged segments saved to: ../result/vlm_inference/InternVL3_5-38B/Task5_InternVL3_5-38B_all_merged.csv
# Processed 371 videos from 2078 segments
# ../result/vlm_inference/Lingshu-32B/Task5_Lingshu-32B_all.csv
# Merged segments saved to: ../result/vlm_inference/Lingshu-32B/Task5_Lingshu-32B_all_merged.csv
# Processed 417 videos from 2380 segments
# ../result/vlm_inference/Qwen2.5-Omni-7B/Task5_Qwen2.5-Omni-7B_all.csv
# Merged segments saved to: ../result/vlm_inference/Qwen2.5-Omni-7B/Task5_Qwen2.5-Omni-7B_all_merged.csv
# Processed 372 videos from 2116 segments
# ../result/vlm_inference/Qwen2.5-VL-7B-Instruct/Task5_Qwen2.5-VL-7B-Instruct_all.csv
# Merged segments saved to: ../result/vlm_inference/Qwen2.5-VL-7B-Instruct/Task5_Qwen2.5-VL-7B-Instruct_all_merged.csv
# Processed 386 videos from 2175 segments
# ../result/vlm_inference/Qwen2.5-VL-32B-Instruct/Task5_Qwen2.5-VL-32B-Instruct_all.csv
# Merged segments saved to: ../result/vlm_inference/Qwen2.5-VL-32B-Instruct/Task5_Qwen2.5-VL-32B-Instruct_all_merged.csv
# Processed 388 videos from 2111 segments
# ../result/vlm_inference/Qwen2.5-VL-72B-Instruct/Task5_Qwen2.5-VL-72B-Instruct_all.csv
# Merged segments saved to: ../result/vlm_inference/Qwen2.5-VL-72B-Instruct/Task5_Qwen2.5-VL-72B-Instruct_all_merged.csv
# Processed 424 videos from 2413 segments