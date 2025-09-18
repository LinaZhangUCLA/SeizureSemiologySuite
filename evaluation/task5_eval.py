#!/usr/bin/env python3
"""
Task 5 Evaluation Script

This script compares prediction and ground truth files containing time periods
and calculates MSE and IOU metrics between the time segments.

File format expected:
file_name,start_seconds,end_seconds
A0002@5-13-2021@UA6693LK@sz_v1_1.mp4,8,110
A0002@5-13-2021@UA6693LK@sz_v2_1.mp4,12,48
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import sys
import os
from typing import Tuple, Dict, List


def calculate_mse(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """
    Calculate Mean Squared Error between predicted and ground truth time periods.
    
    Args:
        pred_start: Predicted start time in seconds
        pred_end: Predicted end time in seconds
        gt_start: Ground truth start time in seconds
        gt_end: Ground truth end time in seconds
    
    Returns:
        MSE value
    """
    start_error = (pred_start - gt_start) ** 2
    end_error = (pred_end - gt_end) ** 2
    return (start_error + end_error) / 2


def calculate_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """
    Calculate Intersection over Union (IoU) between predicted and ground truth time periods.
    
    Args:
        pred_start: Predicted start time in seconds
        pred_end: Predicted end time in seconds
        gt_start: Ground truth start time in seconds
        gt_end: Ground truth end time in seconds
    
    Returns:
        IoU value between 0 and 1
    """
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


def load_and_validate_file(file_path: str) -> pd.DataFrame:
    """
    Load CSV file and validate its format.
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        Validated DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        
        # Check for either format: start_seconds/end_seconds or start_time/end_time
        if 'start_seconds' in df.columns and 'end_seconds' in df.columns:
            start_col, end_col = 'start_seconds', 'end_seconds'
        elif 'start_time' in df.columns and 'end_time' in df.columns:
            start_col, end_col = 'start_time', 'end_time'
            # Rename columns for consistency
            df = df.rename(columns={'start_time': 'start_seconds', 'end_time': 'end_seconds'})
        else:
            raise ValueError(f"Missing required time columns. Expected either ['start_seconds', 'end_seconds'] or ['start_time', 'end_time']. Got: {list(df.columns)}")
        
        # Validate that end_seconds > start_seconds
        invalid_rows = df[df['end_seconds'] <= df['start_seconds']]
        if not invalid_rows.empty:
            print(f"Warning: Found {len(invalid_rows)} rows where end_seconds <= start_seconds")
            print(invalid_rows[['file_name', 'start_seconds', 'end_seconds']])
        
        return df
        
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading file {file_path}: {e}")


def evaluate_predictions(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> Dict:
    """
    Evaluate predictions against ground truth using MSE and IoU metrics.
    
    Args:
        pred_df: Predictions DataFrame
        gt_df: Ground truth DataFrame
    
    Returns:
        Dictionary containing mean MSE and IoU
    """
    # Create dictionaries for faster lookup
    pred_dict = {row['file_name']: row for _, row in pred_df.iterrows()}
    gt_dict = {row['file_name']: row for _, row in gt_df.iterrows()}
    
    mse_scores = []
    iou_scores = []
    
    # Evaluate matched files
    count=0
    for file_name in gt_dict.keys():
        if file_name in pred_dict:
            count+=1
            pred_row = pred_dict[file_name]
            gt_row = gt_dict[file_name]
            
            mse = calculate_mse(
                pred_row['start_seconds'], pred_row['end_seconds'],
                gt_row['start_seconds'], gt_row['end_seconds']
            )
            
            iou = calculate_iou(
                pred_row['start_seconds'], pred_row['end_seconds'],
                gt_row['start_seconds'], gt_row['end_seconds']
            )
            
            mse_scores.append(mse)
            iou_scores.append(iou)
    
    # Calculate and return mean metrics
    print(f"Evaluated {count} files")
    return {
        'mean_mse': np.mean(mse_scores) if mse_scores else 0.0,
        'mean_iou': np.mean(iou_scores) if iou_scores else 0.0
    }


def print_results(results: Dict):
    """Print evaluation results in a formatted way."""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print(f"\nMean Squared Error (MSE): {results['mean_mse']:.4f}")
    print(f"Mean Intersection over Union (IoU): {results['mean_iou']:.4f}")


def calculate_task5_metrics(pred_path: str, gt_path: str) -> Dict:
    """Calculate MSE and IoU metrics for a single model."""
    try:
        pred_df = load_and_validate_file(pred_path)
        gt_df = load_and_validate_file(gt_path)
        
        # Remove rows with invalid data (NaN or empty values)
        pred_df = pred_df.dropna(subset=['file_name', 'start_seconds', 'end_seconds'])
        gt_df = gt_df.dropna(subset=['file_name', 'start_seconds', 'end_seconds'])
        
        results = evaluate_predictions(pred_df, gt_df)
        return results
        
    except Exception as e:
        print(f"Error processing {pred_path}: {e}")
        return {'mean_mse': None, 'mean_iou': None}


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
    
    ground_truth = "result/ground_truth/task5_annotation.csv"
    results_data = []
    
    # Calculate metrics for each model
    for model_name in model_names:
        model_prediction = f"result/vlm_inference/{model_name}/Task5_{model_name}_all_merged.csv"
        print(f"Evaluating {model_name}...")
        
        results = calculate_task5_metrics(model_prediction, ground_truth)
        
        results_data.append({
            'model': model_name,
            'MSE': results['mean_mse'],
            'tIOU': results['mean_iou']
        })
        
        if results['mean_mse'] is not None:
            print(f"  MSE: {results['mean_mse']:.4f}, IoU: {results['mean_iou']:.4f}")
        else:
            print(f"  Failed to calculate metrics")
    
    # Create results DataFrame and save
    results_df = pd.DataFrame(results_data)
    output_path = "metrics/Task5_duration_metrics.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Display results
    print("\nFinal Results:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
