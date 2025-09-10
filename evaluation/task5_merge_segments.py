import pandas as pd
import re
from collections import defaultdict
import sys

def time_to_seconds(time_str):
    """Convert MM:SS format to total seconds"""
    if pd.isna(time_str) or time_str == '':
        return 0
    minutes, seconds = map(int, str(time_str).split(':'))
    return minutes * 60 + seconds

def seconds_to_time(total_seconds):
    """Convert total seconds back to MM:SS format"""
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"

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
        
        # Get start time from first segment (segment_0)
        start_time = segments[0]['start_time']
        
        # Get the last segment number and its end time
        last_segment = segments[-1]
        last_segment_num = int(re.search(r'_segment_(\d+)', last_segment['file_name']).group(1))
        last_segment_end_time = last_segment['end_time']
        
        # Calculate merged end time: (last_segment_num * 25) + end_time_of_last_segment
        last_end_seconds = time_to_seconds(last_segment_end_time)
        
        # Formula: end_time = (last_segment_num * 25) + end_time_of_last_segment
        total_end_seconds = (last_segment_num * 25) + last_end_seconds
        end_time = seconds_to_time(total_end_seconds)
        
        merged_results.append({
            'file_name': base_filename,
            'start_time': start_time,
            'end_time': end_time
        })
    
    # Create output DataFrame and save
    output_df = pd.DataFrame(merged_results)
    output_df.to_csv(output_csv_path, index=False)
    print(f"Merged segments saved to: {output_csv_path}")
    print(f"Processed {len(merged_results)} videos from {len(df)} segments")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python task5_merge_segments.py <input_csv_path> <output_csv_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    merge_segments(input_path, output_path)
