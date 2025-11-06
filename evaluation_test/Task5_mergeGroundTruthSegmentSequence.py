import pandas as pd
import re

def parse_time_to_seconds(time_str):
    """Convert time string to seconds
    Args:
        time_str: Time string in format 'MM:SS:00' or 'MM:SS'
    Returns:
        int: Time in seconds, or None if invalid
    """
    if not isinstance(time_str, str) or not time_str.strip():
        return None
    
    try:
        # Split the string into parts
        parts = time_str.strip().split(':')
        
        # Handle MM:SS:00 or MM:SS format where MM is minutes, SS is seconds
        minutes = int(parts[0])
        seconds = int(parts[1])
        
        if minutes < 0 or seconds < 0 or seconds >= 60:
            return None
            
        total_seconds = minutes * 60 + seconds
        return total_seconds
            
    except (ValueError, IndexError):
        return None


def seconds_to_mmss(seconds):
    """Convert seconds to MM:SS format
    Args:
        seconds: Time in seconds
    Returns:
        str: Time in MM:SS format
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def generate_intervals(duration_seconds):
    """Generate 30-second intervals with 5-second overlap
    Args:
        duration_seconds: Video duration in seconds
    Returns:
        list: List of interval strings in format ["00:00-00:30", "00:25-00:55", ...]
    """
    intervals = []
    start = 0
    
    while start < duration_seconds:
        # Calculate end time (30 seconds from start, but cap at duration)
        end = min(start + 30, duration_seconds)
        
        # Format as MM:SS-MM:SS
        interval_str = f"{seconds_to_mmss(start)}-{seconds_to_mmss(end)}"
        intervals.append(interval_str)
        
        # If we've reached the duration, stop
        if end >= duration_seconds:
            break
            
        # Move to next interval start (25 seconds later for 5-second overlap)
        start += 25
    
    return intervals


def get_video_duration(row, feature_columns):
    """Get the maximum end time for a video across all features
    Args:
        row: DataFrame row containing video annotations
        feature_columns: List of feature column names
    Returns:
        int: Duration in seconds, or None if no valid end times found
    """
    max_end_time = 0
    found_valid_time = False
    
    for feature in feature_columns:
        end_time_col = f'end_time_for_{feature}'
        
        if end_time_col not in row:
            continue
            
        time_str = row[end_time_col]
        seconds = parse_time_to_seconds(str(time_str))
        
        if seconds is not None:
            found_valid_time = True
            max_end_time = max(max_end_time, seconds)
    
    return max_end_time if found_valid_time else None


def get_valid_features(row, feature_columns):
    """Get all features that are marked 'yes' with valid start and end times
    Args:
        row: DataFrame row containing video annotations
        feature_columns: List of feature column names
    Returns:
        list: List of tuples (feature_name, start_time, end_time) sorted by start_time
    """
    valid_features = []
    
    for feature in feature_columns:
        # Check if feature is marked as 'yes'
        if feature not in row or str(row[feature]).strip().lower() != 'yes':
            continue
        
        start_time_col = f'start_time_for_{feature}'
        end_time_col = f'end_time_for_{feature}'
        
        # Check if time columns exist
        if start_time_col not in row or end_time_col not in row:
            continue
        
        # Parse start and end times
        start_time = parse_time_to_seconds(str(row[start_time_col]))
        end_time = parse_time_to_seconds(str(row[end_time_col]))
        
        # Only include if both times are valid
        if start_time is not None and end_time is not None:
            valid_features.append((feature, start_time, end_time))
    
    # Sort by start_time
    valid_features.sort(key=lambda x: x[1])
    
    return valid_features


def get_features_in_interval(valid_features, interval_start, interval_end):
    """Get features that overlap with the given time interval
    Args:
        valid_features: List of tuples (feature_name, start_time, end_time)
        interval_start: Interval start time in seconds
        interval_end: Interval end time in seconds
    Returns:
        list: List of feature names that overlap with the interval, sorted by start_time
    """
    overlapping_features = []
    
    for feature_name, start_time, end_time in valid_features:
        # Check if feature overlaps with interval
        # Overlap condition: start_time <= interval_end AND end_time >= interval_start
        if start_time <= interval_end and end_time >= interval_start:
            overlapping_features.append(feature_name)
    
    return overlapping_features


def main():
    # Input and output paths
    input_path = 'result/ground_truth/task12_annotation.csv'
    output_path = 'result/ground_truth/task5_segment_sequence_annotation.csv'
    output_path_vlm = 'result/ground_truth/task5_segment_sequence_annotation_vlm.csv'
    
    # Features to exclude from VLM output
    vlm_exclude_features = ['verbal_responsiveness', 'ictal_vocalization']
    
    # Read input CSV
    df = pd.read_csv(input_path, encoding='latin-1')
    
    # Filter out 'nan' entries
    df = df[df['file_name'].notna()]
    print(f"Total valid videos in input: {len(df)}")
    
    # Get feature columns (exclude metadata and justification columns)
    feature_columns = [col for col in df.columns 
                      if not any(x in col for x in ['file_name', 'justification', 'start_time', 'end_time'])]
    
    # Exclude "occur_during_sleep" since it has no start_time column
    feature_columns = [col for col in feature_columns if col != 'occur_during_sleep']
    
    print(f"Found {len(feature_columns)} features")
    
    # Process each video
    output_data = []
    output_data_vlm = []
    videos_without_duration = []
    total_segments = 0
    
    for idx, row in df.iterrows():
        file_name = row['file_name']
        
        # Remove .mp4 extension to prepare for segment naming
        base_name = file_name.replace('.mp4', '')
        
        # Get video duration
        duration = get_video_duration(row, feature_columns)
        
        if duration is None or duration == 0:
            videos_without_duration.append(file_name)
            print(f"Warning: No valid end times found for {file_name}")
            continue
        
        # Get all valid features for this video (sorted by start_time)
        valid_features = get_valid_features(row, feature_columns)
        
        # Generate intervals
        intervals = generate_intervals(duration)
        
        # Process each interval
        for i, interval_str in enumerate(intervals):
            # Parse interval start and end times
            interval_parts = interval_str.split('-')
            interval_start = parse_time_to_seconds(interval_parts[0])
            interval_end = parse_time_to_seconds(interval_parts[1])
            
            # Get features that overlap with this interval
            features_in_interval = get_features_in_interval(valid_features, interval_start, interval_end)
            
            # Create segment video name
            segment_video_name = f"{base_name}_segment_{i}.mp4"
            
            # Create segment feature list for regular output (comma-separated)
            if features_in_interval:
                segment_feature_list = '"' + ', '.join(features_in_interval) + '"'
            else:
                segment_feature_list = ""
            
            output_data.append({
                'segment_video_name': segment_video_name,
                'segment_feature_list': segment_feature_list
            })
            
            # Create segment feature list for VLM output (exclude acoustic features)
            features_in_interval_vlm = [f for f in features_in_interval if f not in vlm_exclude_features]
            if features_in_interval_vlm:
                segment_feature_list_vlm = '"' + ', '.join(features_in_interval_vlm) + '"'
            else:
                segment_feature_list_vlm = ""
            
            output_data_vlm.append({
                'segment_video_name': segment_video_name,
                'segment_feature_list': segment_feature_list_vlm
            })
            
            total_segments += 1
    
    # Print summary
    print(f"\nSummary:")
    print(f"Total videos processed: {len(df)}")
    print(f"Videos with valid duration: {len(df) - len(videos_without_duration)}")
    print(f"Videos without valid duration: {len(videos_without_duration)}")
    print(f"Total segments generated: {total_segments}")
    
    # Show example output
    if output_data:
        print(f"\nExample output (first 3 segments - regular):")
        for i in range(min(3, len(output_data))):
            print(f"  Segment: {output_data[i]['segment_video_name']}")
            print(f"  Features: {output_data[i]['segment_feature_list']}")
            print()
        
        print(f"Example output (first 3 segments - VLM, excluding {', '.join(vlm_exclude_features)}):")
        for i in range(min(3, len(output_data_vlm))):
            print(f"  Segment: {output_data_vlm[i]['segment_video_name']}")
            print(f"  Features: {output_data_vlm[i]['segment_feature_list']}")
            print()
    
    # Save output file manually to avoid quote issues
    def write_csv_without_quotes(data, filepath):
        """Write CSV with no quotes on any values"""
        with open(filepath, 'w', encoding='utf-8') as f:
            # Write header
            f.write('segment_video_name,segment_feature_list\n')
            # Write data rows
            for row in data:
                f.write(f'{row["segment_video_name"]},{row["segment_feature_list"]}\n')
    
    write_csv_without_quotes(output_data, output_path)
    print(f"\nSaved segment feature sequences to {output_path}")
    
    write_csv_without_quotes(output_data_vlm, output_path_vlm)
    print(f"Saved VLM segment feature sequences (excluding {', '.join(vlm_exclude_features)}) to {output_path_vlm}")


if __name__ == "__main__":
    main()

