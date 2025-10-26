#!/usr/bin/env python3
"""
Task 5 Evaluation Script

This script compares prediction and ground truth files containing time periods
and calculates MAE and IOU metrics between the time segments.

Prediction file format:
video_name,feature,timestamp
N0017@12-4-2018@DA7021YP@sz_v1_1_blank_stare.mp4,blank_stare,00:00

Ground truth file format (multiple CSV files in a directory):
file_name,target_time
A0003@10-20-2020@UA6692Q1@sz_v2_1_blank_stare.mp4,30
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import sys
import os
from typing import Tuple, Dict, List
import glob


def timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert MM:SS timestamp to seconds.
    
    Args:
        timestamp: Time string in MM:SS format
    
    Returns:
        Time in seconds
    """
    try:
        parts = timestamp.strip().split(':')
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + int(seconds)
        elif len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        else:
            raise ValueError(f"Invalid timestamp format: {timestamp}")
    except Exception as e:
        raise ValueError(f"Error parsing timestamp '{timestamp}': {e}")


def calculate_mae(pred_time: float, gt_time: float) -> float:
    """
    Calculate Mean Absolute Error between predicted and ground truth time.
    
    Args:
        pred_time: Predicted time in seconds
        gt_time: Ground truth time in seconds
    
    Returns:
        MAE value (absolute difference)
    """
    return abs(pred_time - gt_time)


def calculate_iou(pred_time: float, gt_time: float, tolerance: float = 5.0) -> float:
    """
    Calculate IoU-like metric for time points.
    Since we're comparing time points, we use a tolerance window.
    
    Args:
        pred_time: Predicted time in seconds
        gt_time: Ground truth time in seconds
        tolerance: Tolerance window in seconds (default 5 seconds)
    
    Returns:
        IoU-like value between 0 and 1
    """
    # Create windows around each time point
    pred_start, pred_end = pred_time - tolerance, pred_time + tolerance
    gt_start, gt_end = gt_time - tolerance, gt_time + tolerance
    
    # Calculate intersection
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    intersection = max(0, intersection_end - intersection_start)
    
    # Calculate union
    union_start = min(pred_start, gt_start)
    union_end = max(pred_end, gt_end)
    union = union_end - union_start
    
    # Avoid division by zero
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def load_prediction_file(file_path: str) -> pd.DataFrame:
    """
    Load prediction CSV file with new format.
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        DataFrame with file_name, feature, timestamp columns
    """
    try:
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_cols = ['video_name', 'feature', 'timestamp']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Expected {required_cols}. Got: {list(df.columns)}")
        
        # Process each row
        processed_rows = []
        for _, row in df.iterrows():
            video_name = row['video_name']
            feature = row['feature']
            timestamp = row['timestamp']
            
            # Convert timestamp to seconds
            time_seconds = timestamp_to_seconds(timestamp)
            
            processed_rows.append({
                'file_name': video_name,
                'feature': feature,
                'time_seconds': time_seconds
            })
        
        result_df = pd.DataFrame(processed_rows)
        return result_df
        
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading file {file_path}: {e}")


def load_ground_truth_directory(dir_path: str) -> pd.DataFrame:
    """
    Load all ground truth CSV files from a directory.
    
    Args:
        dir_path: Path to the directory containing ground truth CSV files
    
    Returns:
        Combined DataFrame with file_name, feature, target_time columns
    """
    try:
        if not os.path.isdir(dir_path):
            raise ValueError(f"Ground truth path is not a directory: {dir_path}")
        
        # Find all CSV files in the directory
        csv_files = glob.glob(os.path.join(dir_path, "*.csv"))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {dir_path}")
        
        all_data = []
        
        for csv_file in csv_files:
            # Extract feature name from filename (e.g., blank_stare_timestamps.csv -> blank_stare)
            filename = os.path.basename(csv_file)
            feature = filename.replace('_timestamps.csv', '')
            
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Validate columns
            if 'file_name' not in df.columns or 'target_time' not in df.columns:
                print(f"Warning: Skipping {csv_file} - missing required columns")
                continue
            
            # Add feature column
            df['feature'] = feature
            df = df.rename(columns={'target_time': 'time_seconds'})
            
            all_data.append(df[['file_name', 'feature', 'time_seconds']])
        
        # Combine all dataframes
        if not all_data:
            raise ValueError("No valid ground truth data found")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(combined_df)} ground truth entries from {len(csv_files)} files")
        
        return combined_df
        
    except Exception as e:
        raise Exception(f"Error loading ground truth from {dir_path}: {e}")


def evaluate_predictions(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> Dict:
    """
    Evaluate predictions against ground truth using MAE and IoU metrics.
    
    Args:
        pred_df: Predictions DataFrame
        gt_df: Ground truth DataFrame
    
    Returns:
        Dictionary containing mean MAE and IoU
    """
    # Create dictionaries for faster lookup (key: file_name)
    pred_dict = {}
    for _, row in pred_df.iterrows():
        key = row['file_name']
        pred_dict[key] = row
    
    gt_dict = {}
    for _, row in gt_df.iterrows():
        key = row['file_name']
        gt_dict[key] = row
    
    mae_scores = []
    iou_scores = []
    
    # Evaluate matched files
    count = 0
    matched_files = []
    unmatched_gt_files = []
    unmatched_pred_files = []
    
    for file_name in gt_dict.keys():
        if file_name in pred_dict:
            count += 1
            matched_files.append(file_name)
            pred_row = pred_dict[file_name]
            gt_row = gt_dict[file_name]
            
            mae = calculate_mae(pred_row['time_seconds'], gt_row['time_seconds'])
            iou = calculate_iou(pred_row['time_seconds'], gt_row['time_seconds'])
            
            mae_scores.append(mae)
            iou_scores.append(iou)
        else:
            unmatched_gt_files.append(file_name)
    
    # Find predictions without ground truth
    for file_name in pred_dict.keys():
        if file_name not in gt_dict:
            unmatched_pred_files.append(file_name)
    
    # Print matching statistics
    print(f"Evaluated {count} files")
    print(f"Ground truth files: {len(gt_dict)}, Prediction files: {len(pred_dict)}")
    print(f"Matched: {count}, Unmatched GT: {len(unmatched_gt_files)}, Unmatched Pred: {len(unmatched_pred_files)}")
    
    if unmatched_gt_files:
        print(f"First few unmatched GT files: {unmatched_gt_files[:3]}")
    if unmatched_pred_files:
        print(f"First few unmatched Pred files: {unmatched_pred_files[:3]}")
    
    # Calculate and return mean metrics
    return {
        'mean_mae': np.mean(mae_scores) if mae_scores else 0.0,
        'mean_iou': np.mean(iou_scores) if iou_scores else 0.0,
        'matched_count': count,
        'total_gt': len(gt_dict),
        'total_pred': len(pred_dict)
    }


def print_results(results: Dict):
    """Print evaluation results in a formatted way."""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print(f"\nMean Absolute Error (MAE): {results['mean_mae']:.4f}")
    print(f"Mean Intersection over Union (IoU): {results['mean_iou']:.4f}")


def calculate_task5_metrics(pred_path: str, gt_path: str) -> Dict:
    """Calculate MAE and IoU metrics for a single model."""
    try:
        pred_df = load_prediction_file(pred_path)
        gt_df = load_ground_truth_directory(gt_path)
        
        # Remove rows with invalid data (NaN or empty values)
        pred_df = pred_df.dropna(subset=['file_name', 'time_seconds'])
        gt_df = gt_df.dropna(subset=['file_name', 'time_seconds'])
        
        results = evaluate_predictions(pred_df, gt_df)
        return results
        
    except Exception as e:
        print(f"Error processing {pred_path}: {e}")
        import traceback
        traceback.print_exc()
        return {'mean_mae': None, 'mean_iou': None}


def main():
    """Main function to evaluate all models and save results."""
    # Change to SeizureSemiologyBench directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    # Model names to evaluate
    model_names = [
        "InternVL3_5-8B",
        "InternVL3_5-38B", 
        "Lingshu-32B",
        "Qwen2.5-Omni-7B",
        "Qwen2.5-VL-7B-Instruct",
        "Qwen2.5-VL-32B-Instruct",
        "Qwen2.5-VL-72B-Instruct"
    ]
    
    ground_truth = "result/ground_truth/task4_groundtruth"
    results_data = []
    
    # Calculate metrics for each model
    for model_name in model_names:
        model_prediction = f"result/vlm_inference/{model_name}/Task4_{model_name}_0.csv"
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}...")
        print(f"{'='*60}")
        
        results = calculate_task5_metrics(model_prediction, ground_truth)
        
        results_data.append({
            'model': model_name,
            'MAE': results['mean_mae'],
            'tIOU': results['mean_iou']
        })
        
        if results['mean_mae'] is not None:
            print(f"  MAE: {results['mean_mae']:.4f}, IoU: {results['mean_iou']:.4f}")
        else:
            print(f"  Failed to calculate metrics")
    
    # Create results DataFrame and save
    results_df = pd.DataFrame(results_data)
    output_path = "metrics/Task4_time_metrics.csv"
    
    # Create metrics directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    results_df.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    # Display results
    print("\nFinal Results:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()

# change mae mertrics, but do not change input csv format
# #!/usr/bin/env python3
# """
# Task 5 Evaluation Script

# This script compares prediction and ground truth files containing time periods
# and calculates MAE and IOU metrics between the time segments.

# File format expected:
# file_name,start_seconds,end_seconds
# A0002@5-13-2021@UA6693LK@sz_v1_1.mp4,8,110
# A0002@5-13-2021@UA6693LK@sz_v2_1.mp4,12,48
# """

# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import argparse
# import sys
# import os
# from typing import Tuple, Dict, List


# def calculate_mae(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
#     """
#     Calculate Mean Absolute Error between predicted and ground truth time periods.
    
#     Args:
#         pred_start: Predicted start time in seconds
#         pred_end: Predicted end time in seconds
#         gt_start: Ground truth start time in seconds
#         gt_end: Ground truth end time in seconds
    
#     Returns:
#         MAE value
#     """
#     start_error = abs(pred_start - gt_start)
#     end_error = abs(pred_end - gt_end)
#     return (start_error + end_error) / 2


# def calculate_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
#     """
#     Calculate Intersection over Union (IoU) between predicted and ground truth time periods.
    
#     Args:
#         pred_start: Predicted start time in seconds
#         pred_end: Predicted end time in seconds
#         gt_start: Ground truth start time in seconds
#         gt_end: Ground truth end time in seconds
    
#     Returns:
#         IoU value between 0 and 1
#     """
#     # Calculate intersection
#     intersection_start = max(pred_start, gt_start)
#     intersection_end = min(pred_end, gt_end)
#     intersection = max(0, intersection_end - intersection_start)
    
#     # Calculate union
#     union_start = min(pred_start, gt_start)
#     union_end = max(pred_end, gt_end)
#     union = union_end - union_start
    
#     # Avoid division by zero
#     if union == 0:
#         return 1.0 if intersection == 0 else 0.0
    
#     return intersection / union


# def load_and_validate_file(file_path: str) -> pd.DataFrame:
#     """
#     Load CSV file and validate its format.
    
#     Args:
#         file_path: Path to the CSV file
    
#     Returns:
#         Validated DataFrame
#     """
#     try:
#         df = pd.read_csv(file_path)
        
#         # Check for either format: start_seconds/end_seconds or start_time/end_time
#         if 'start_seconds' in df.columns and 'end_seconds' in df.columns:
#             start_col, end_col = 'start_seconds', 'end_seconds'
#         elif 'start_time' in df.columns and 'end_time' in df.columns:
#             start_col, end_col = 'start_time', 'end_time'
#             # Rename columns for consistency
#             df = df.rename(columns={'start_time': 'start_seconds', 'end_time': 'end_seconds'})
#         else:
#             raise ValueError(f"Missing required time columns. Expected either ['start_seconds', 'end_seconds'] or ['start_time', 'end_time']. Got: {list(df.columns)}")
        
#         # Validate that end_seconds > start_seconds
#         invalid_rows = df[df['end_seconds'] <= df['start_seconds']]
#         if not invalid_rows.empty:
#             print(f"Warning: Found {len(invalid_rows)} rows where end_seconds <= start_seconds")
#             print(invalid_rows[['file_name', 'start_seconds', 'end_seconds']])
        
#         return df
        
#     except FileNotFoundError:
#         raise FileNotFoundError(f"File not found: {file_path}")
#     except Exception as e:
#         raise Exception(f"Error loading file {file_path}: {e}")


# def evaluate_predictions(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> Dict:
#     """
#     Evaluate predictions against ground truth using MAE and IoU metrics.
    
#     Args:
#         pred_df: Predictions DataFrame
#         gt_df: Ground truth DataFrame
    
#     Returns:
#         Dictionary containing mean MAE and IoU
#     """
#     # Create dictionaries for faster lookup
#     pred_dict = {row['file_name']: row for _, row in pred_df.iterrows()}
#     gt_dict = {row['file_name']: row for _, row in gt_df.iterrows()}
    
#     mae_scores = []
#     iou_scores = []
    
#     # Evaluate matched files
#     count=0
#     for file_name in gt_dict.keys():
#         if file_name in pred_dict:
#             count+=1
#             pred_row = pred_dict[file_name]
#             gt_row = gt_dict[file_name]
            
#             mae = calculate_mae(
#                 pred_row['start_seconds'], pred_row['end_seconds'],
#                 gt_row['start_seconds'], gt_row['end_seconds']
#             )
            
#             iou = calculate_iou(
#                 pred_row['start_seconds'], pred_row['end_seconds'],
#                 gt_row['start_seconds'], gt_row['end_seconds']
#             )
            
#             mae_scores.append(mae)
#             iou_scores.append(iou)
    
#     # Calculate and return mean metrics
#     print(f"Evaluated {count} files")
#     return {
#         'mean_mae': np.mean(mae_scores) if mae_scores else 0.0,
#         'mean_iou': np.mean(iou_scores) if iou_scores else 0.0
#     }


# def print_results(results: Dict):
#     """Print evaluation results in a formatted way."""
#     print("\n" + "="*50)
#     print("EVALUATION RESULTS")
#     print("="*50)
    
#     print(f"\nMean Absolute Error (MAE): {results['mean_mae']:.4f}")
#     print(f"Mean Intersection over Union (IoU): {results['mean_iou']:.4f}")


# def calculate_task5_metrics(pred_path: str, gt_path: str) -> Dict:
#     """Calculate MAE and IoU metrics for a single model."""
#     try:
#         pred_df = load_and_validate_file(pred_path)
#         gt_df = load_and_validate_file(gt_path)
        
#         # Remove rows with invalid data (NaN or empty values)
#         pred_df = pred_df.dropna(subset=['file_name', 'start_seconds', 'end_seconds'])
#         gt_df = gt_df.dropna(subset=['file_name', 'start_seconds', 'end_seconds'])
        
#         results = evaluate_predictions(pred_df, gt_df)
#         return results
        
#     except Exception as e:
#         print(f"Error processing {pred_path}: {e}")
#         return {'mean_mae': None, 'mean_iou': None}


# def main():
#     """Main function to evaluate all models and save results."""
#     # Change to SeizureSemiologyBench directory
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     project_root = os.path.dirname(script_dir)
#     os.chdir(project_root)
    
#     # Model names to evaluate
#     model_names = [
#         "InternVL3_5-8B",
#         "InternVL3_5-38B", 
#         "Lingshu-32B",
#         "Qwen2.5-Omni-7B",
#         "Qwen2.5-VL-7B-Instruct",
#         "Qwen2.5-VL-32B-Instruct",
#         "Qwen2.5-VL-72B-Instruct"
#     ]
    
#     ground_truth = "result/ground_truth/task4_groundtruth"
#     results_data = []
    
#     # Calculate metrics for each model
#     for model_name in model_names:
#         model_prediction = f"result/vlm_inference/{model_name}/Task4_{model_name}_0.csv"
#         print(f"Evaluating {model_name}...")
        
#         results = calculate_task5_metrics(model_prediction, ground_truth)
        
#         results_data.append({
#             'model': model_name,
#             'MAE': results['mean_mae'],
#             'tIOU': results['mean_iou']
#         })
        
#         if results['mean_mae'] is not None:
#             print(f"  MAE: {results['mean_mae']:.4f}, IoU: {results['mean_iou']:.4f}")
#         else:
#             print(f"  Failed to calculate metrics")
    
#     # Create results DataFrame and save
#     results_df = pd.DataFrame(results_data)
#     output_path = "metrics/Task4_time_metrics.csv"
#     results_df.to_csv(output_path, index=False)
#     print(f"\nResults saved to: {output_path}")
    
#     # Display results
#     print("\nFinal Results:")
#     print(results_df.to_string(index=False))


# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# """
# Task 5 Evaluation Script

# This script compares prediction and ground truth files containing time periods
# and calculates MSE and IOU metrics between the time segments.

# File format expected:
# file_name,start_seconds,end_seconds
# A0002@5-13-2021@UA6693LK@sz_v1_1.mp4,8,110
# A0002@5-13-2021@UA6693LK@sz_v2_1.mp4,12,48
# """

# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import argparse
# import sys
# import os
# from typing import Tuple, Dict, List


# def calculate_mse(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
#     """
#     Calculate Mean Squared Error between predicted and ground truth time periods.
    
#     Args:
#         pred_start: Predicted start time in seconds
#         pred_end: Predicted end time in seconds
#         gt_start: Ground truth start time in seconds
#         gt_end: Ground truth end time in seconds
    
#     Returns:
#         MSE value
#     """
#     start_error = (pred_start - gt_start) ** 2
#     end_error = (pred_end - gt_end) ** 2
#     return (start_error + end_error) / 2


# def calculate_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
#     """
#     Calculate Intersection over Union (IoU) between predicted and ground truth time periods.
    
#     Args:
#         pred_start: Predicted start time in seconds
#         pred_end: Predicted end time in seconds
#         gt_start: Ground truth start time in seconds
#         gt_end: Ground truth end time in seconds
    
#     Returns:
#         IoU value between 0 and 1
#     """
#     # Calculate intersection
#     intersection_start = max(pred_start, gt_start)
#     intersection_end = min(pred_end, gt_end)
#     intersection = max(0, intersection_end - intersection_start)
    
#     # Calculate union
#     union_start = min(pred_start, gt_start)
#     union_end = max(pred_end, gt_end)
#     union = union_end - union_start
    
#     # Avoid division by zero
#     if union == 0:
#         return 1.0 if intersection == 0 else 0.0
    
#     return intersection / union


# def load_and_validate_file(file_path: str) -> pd.DataFrame:
#     """
#     Load CSV file and validate its format.
    
#     Args:
#         file_path: Path to the CSV file
    
#     Returns:
#         Validated DataFrame
#     """
#     try:
#         df = pd.read_csv(file_path)
        
#         # Check for either format: start_seconds/end_seconds or start_time/end_time
#         if 'start_seconds' in df.columns and 'end_seconds' in df.columns:
#             start_col, end_col = 'start_seconds', 'end_seconds'
#         elif 'start_time' in df.columns and 'end_time' in df.columns:
#             start_col, end_col = 'start_time', 'end_time'
#             # Rename columns for consistency
#             df = df.rename(columns={'start_time': 'start_seconds', 'end_time': 'end_seconds'})
#         else:
#             raise ValueError(f"Missing required time columns. Expected either ['start_seconds', 'end_seconds'] or ['start_time', 'end_time']. Got: {list(df.columns)}")
        
#         # Validate that end_seconds > start_seconds
#         invalid_rows = df[df['end_seconds'] <= df['start_seconds']]
#         if not invalid_rows.empty:
#             print(f"Warning: Found {len(invalid_rows)} rows where end_seconds <= start_seconds")
#             print(invalid_rows[['file_name', 'start_seconds', 'end_seconds']])
        
#         return df
        
#     except FileNotFoundError:
#         raise FileNotFoundError(f"File not found: {file_path}")
#     except Exception as e:
#         raise Exception(f"Error loading file {file_path}: {e}")


# def evaluate_predictions(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> Dict:
#     """
#     Evaluate predictions against ground truth using MSE and IoU metrics.
    
#     Args:
#         pred_df: Predictions DataFrame
#         gt_df: Ground truth DataFrame
    
#     Returns:
#         Dictionary containing mean MSE and IoU
#     """
#     # Create dictionaries for faster lookup
#     pred_dict = {row['file_name']: row for _, row in pred_df.iterrows()}
#     gt_dict = {row['file_name']: row for _, row in gt_df.iterrows()}
    
#     mse_scores = []
#     iou_scores = []
    
#     # Evaluate matched files
#     count=0
#     for file_name in gt_dict.keys():
#         if file_name in pred_dict:
#             count+=1
#             pred_row = pred_dict[file_name]
#             gt_row = gt_dict[file_name]
            
#             mse = calculate_mse(
#                 pred_row['start_seconds'], pred_row['end_seconds'],
#                 gt_row['start_seconds'], gt_row['end_seconds']
#             )
            
#             iou = calculate_iou(
#                 pred_row['start_seconds'], pred_row['end_seconds'],
#                 gt_row['start_seconds'], gt_row['end_seconds']
#             )
            
#             mse_scores.append(mse)
#             iou_scores.append(iou)
    
#     # Calculate and return mean metrics
#     print(f"Evaluated {count} files")
#     return {
#         'mean_mse': np.mean(mse_scores) if mse_scores else 0.0,
#         'mean_iou': np.mean(iou_scores) if iou_scores else 0.0
#     }


# def print_results(results: Dict):
#     """Print evaluation results in a formatted way."""
#     print("\n" + "="*50)
#     print("EVALUATION RESULTS")
#     print("="*50)
    
#     print(f"\nMean Squared Error (MSE): {results['mean_mse']:.4f}")
#     print(f"Mean Intersection over Union (IoU): {results['mean_iou']:.4f}")


# def calculate_task5_metrics(pred_path: str, gt_path: str) -> Dict:
#     """Calculate MSE and IoU metrics for a single model."""
#     try:
#         pred_df = load_and_validate_file(pred_path)
#         gt_df = load_and_validate_file(gt_path)
        
#         # Remove rows with invalid data (NaN or empty values)
#         pred_df = pred_df.dropna(subset=['file_name', 'start_seconds', 'end_seconds'])
#         gt_df = gt_df.dropna(subset=['file_name', 'start_seconds', 'end_seconds'])
        
#         results = evaluate_predictions(pred_df, gt_df)
#         return results
        
#     except Exception as e:
#         print(f"Error processing {pred_path}: {e}")
#         return {'mean_mse': None, 'mean_iou': None}


# def main():
#     """Main function to evaluate all models and save results."""
#     # Change to SeizureSemiologyBench directory
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     project_root = os.path.dirname(script_dir)
#     os.chdir(project_root)
    
#     # Model names to evaluate
#     model_names = [
#         "InternVL3_5-8B",
#         "InternVL3_5-38B", 
#         "Lingshu-32B",
#         "Qwen2.5-Omni-7B",
#         "Qwen2.5-VL-7B-Instruct",
#         "Qwen2.5-VL-32B-Instruct",
#         "Qwen2.5-VL-72B-Instruct"
#     ]
    
#     ground_truth = "result/ground_truth/task5_annotation.csv"
#     results_data = []
    
#     # Calculate metrics for each model
#     for model_name in model_names:
#         model_prediction = f"result/vlm_inference/{model_name}/Task5_{model_name}_all_merged.csv"
#         print(f"Evaluating {model_name}...")
        
#         results = calculate_task5_metrics(model_prediction, ground_truth)
        
#         results_data.append({
#             'model': model_name,
#             'MSE': results['mean_mse'],
#             'tIOU': results['mean_iou']
#         })
        
#         if results['mean_mse'] is not None:
#             print(f"  MSE: {results['mean_mse']:.4f}, IoU: {results['mean_iou']:.4f}")
#         else:
#             print(f"  Failed to calculate metrics")
    
#     # Create results DataFrame and save
#     results_df = pd.DataFrame(results_data)
#     output_path = "metrics/Task5_duration_metrics.csv"
#     results_df.to_csv(output_path, index=False)
#     print(f"\nResults saved to: {output_path}")
    
#     # Display results
#     print("\nFinal Results:")
#     print(results_df.to_string(index=False))


# if __name__ == "__main__":
#     main()
