import pandas as pd
import numpy as np
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def extract_label(label_str):
    """Extract ES or NES from label string
    Args:
        label_str: String that may contain ES or NES
    Returns:
        str: 'ES' or 'NES' or None
    """
    if pd.isna(label_str):
        return None
    
    label_str = str(label_str).strip().upper()
    
    # Check for NES first (since NES contains ES substring)
    if 'NES' in label_str:
        return 'NES'
    
    # Then check for ES
    if 'ES' in label_str:
        return 'ES'
    
    return None

def calculate_metrics(y_true, y_pred):
    """Calculate binary classification metrics with ES as positive class
    Args:
        y_true: Ground truth labels (ES/NES)
        y_pred: Predicted labels (ES/NES)
    Returns:
        tuple: (accuracy, precision, recall, f1)
    """
    # Extract and normalize labels
    y_true = y_true.apply(extract_label)
    y_pred = y_pred.apply(extract_label)
    
    # Filter out any invalid values
    valid_values = ['ES', 'NES']
    valid_mask = y_true.isin(valid_values) & y_pred.isin(valid_values)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) == 0 or len(y_pred) == 0:
        return '', '', '', ''
    
    # Binary classification metrics with ES as positive class
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label='ES', zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label='ES', zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label='ES', zero_division=0)
    
    return accuracy, precision, recall, f1

def get_model_metrics(model_name, model_path_mapping):
    """Calculate metrics for a specific model
    Args:
        model_name (str): Name of the model to evaluate
        model_path_mapping (dict): Mapping from display model names to actual file path names
    Returns:
        dict: Dictionary containing metrics for with_report and without_report
    """
    # File paths
    gt_path = 'result/ground_truth/task7_annotation.csv'
    
    # Get actual path name from mapping, or use model_name if not in mapping
    actual_model_name = model_path_mapping.get(model_name, model_name)
    
    pred_path = f'result/vlm_inference/{actual_model_name}/Task7_{actual_model_name}_all.csv'
    
    try:
        # Read ground truth and prediction files
        try:
            df_gt = pd.read_csv(gt_path, encoding='latin-1')
        except FileNotFoundError:
            print(f"Ground truth file not found: {gt_path}")
            # Return empty results to keep model in output with blank metrics
            return {
                'with_report': {'precision': '', 'recall': '', 'f1': '', 'accuracy': ''},
                'without_report': {'precision': '', 'recall': '', 'f1': '', 'accuracy': ''}
            }
        
        try:
            df_pred = pd.read_csv(pred_path, encoding='latin-1')
        except FileNotFoundError:
            print(f"Prediction file not found: {pred_path}")
            # Return empty results to keep model in output with blank metrics
            return {
                'with_report': {'precision': '', 'recall': '', 'f1': '', 'accuracy': ''},
                'without_report': {'precision': '', 'recall': '', 'f1': '', 'accuracy': ''}
            }
        
        # Merge ground truth and predictions
        merged_df = pd.merge(df_gt, df_pred, on='file_name', how='inner')
        
        # Initialize results dictionary
        results = {}
        
        # Calculate metrics for prediction_with_report
        try:
            y_true = merged_df['label']
            y_pred = merged_df['prediction_with_report']
            
            accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred)
            
            results['with_report'] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            }
            
            print(f"Processed with_report - Precision: {float(precision):.2f}, Recall: {float(recall):.2f}, F1: {float(f1):.2f}, Accuracy: {float(accuracy):.2f}")
            
        except Exception as e:
            print(f"Error calculating metrics for with_report: {str(e)}")
            results['with_report'] = {
                'precision': '',
                'recall': '',
                'f1': '',
                'accuracy': ''
            }
        
        # Calculate metrics for prediction_without_report
        try:
            y_true = merged_df['label']
            y_pred = merged_df['prediction_without_report']
            
            accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred)
            
            results['without_report'] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            }
            
            print(f"Processed without_report - Precision: {float(precision):.2f}, Recall: {float(recall):.2f}, F1: {float(f1):.2f}, Accuracy: {float(accuracy):.2f}")
            
        except Exception as e:
            print(f"Error calculating metrics for without_report: {str(e)}")
            results['without_report'] = {
                'precision': '',
                'recall': '',
                'f1': '',
                'accuracy': ''
            }
        
        return results
    
    except Exception as e:
        print(f"Unexpected error processing model {model_name}: {str(e)}")
        # Return empty results to keep model in output with blank metrics
        return {
            'with_report': {'precision': '', 'recall': '', 'f1': '', 'accuracy': ''},
            'without_report': {'precision': '', 'recall': '', 'f1': '', 'accuracy': ''}
        }

def write_metrics_file(model_names, results_dict, output_path):
    """Write metrics to CSV file
    Args:
        model_names (list): List of model names
        results_dict (dict): Dictionary containing metrics for each model
        output_path (str): Path to output file
    """
    try:
        with open(output_path, 'w') as f:
            # Create header row
            header = [
                'model',
                'precision_with_report',
                'recall_with_report',
                'f1_with_report',
                'accuracy_with_report',
                'precision_without_report',
                'recall_without_report',
                'f1_without_report',
                'accuracy_without_report'
            ]
            f.write(','.join(header) + '\n')
            
            # Write data rows
            for model in model_names:
                row = [model]
                
                # Check if model has results (should always exist now)
                if model in results_dict and results_dict[model]:
                    # Metrics with report
                    if 'with_report' in results_dict[model]:
                        metrics = results_dict[model]['with_report']
                        row.extend([
                            f"{float(metrics['precision']):.2f}" if metrics['precision'] != '' else '',
                            f"{float(metrics['recall']):.2f}" if metrics['recall'] != '' else '',
                            f"{float(metrics['f1']):.2f}" if metrics['f1'] != '' else '',
                            f"{float(metrics['accuracy']):.2f}" if metrics['accuracy'] != '' else ''
                        ])
                    else:
                        row.extend(['', '', '', ''])
                    
                    # Metrics without report
                    if 'without_report' in results_dict[model]:
                        metrics = results_dict[model]['without_report']
                        row.extend([
                            f"{float(metrics['precision']):.2f}" if metrics['precision'] != '' else '',
                            f"{float(metrics['recall']):.2f}" if metrics['recall'] != '' else '',
                            f"{float(metrics['f1']):.2f}" if metrics['f1'] != '' else '',
                            f"{float(metrics['accuracy']):.2f}" if metrics['accuracy'] != '' else ''
                        ])
                    else:
                        row.extend(['', '', '', ''])
                else:
                    # If model somehow not in dict, add empty metrics
                    row.extend(['', '', '', '', '', '', '', ''])
                
                f.write(','.join(row) + '\n')
        
        print(f"\nMetrics saved to {output_path}")
    
    except Exception as e:
        print(f"Error writing metrics file: {str(e)}")
        raise

def main():
    # Model names to evaluate (same order as Task1)
    model_list = [
        'Qwen2.5-VL-7B',
        'InternVL3.5-8B',
        'Qwen3-VL-8B-Instruct',
        'Qwen2.5-VL-32B',
        'InternVL3.5-38B',
        'Qwen3-VL-32B-Instruct',
        'Qwen2.5-VL-72B',
        'Audio-flamingo-3',
        'Qwen2.5-Omni-7B',
        "Qwen3-Omni-30B-A3B-Instruct",
        'Lingshu-32B'

        # "InternVL3_5-8B",
        # "Qwen2.5-VL-7B-Instruct",
        # 'Qwen3-VL-8B-Instruct',
        # #"InternVL3_5-38B",
        # "Qwen2.5-VL-32B-Instruct",
        # 'Qwen3-VL-32B-Instruct',
        # "Qwen2.5-VL-72B-Instruct",
        
        # "Qwen2.5-Omni-7B",
        # "Qwen3-Omni-30B-A3B-Instruct",
        # 'Lingshu-32B'
    ]
    
    # Mapping from display model names to actual file path names
    model_path_mapping = {
        'Qwen2.5-VL-7B': 'Qwen2.5-VL-7B-Instruct',
        'InternVL3.5-8B': 'InternVL3_5-8B',
        'Qwen2.5-VL-32B': 'Qwen2.5-VL-32B-Instruct',
        'InternVL3.5-38B': 'InternVL3_5-38B',
        'Qwen2.5-VL-72B': 'Qwen2.5-VL-72B-Instruct',
        'Audio-flamingo-3': 'audio-flamingo-3'
    }
    
    output_path = 'metrics/Task7_precision_recall_f1_accuracy.csv'
    
    # Calculate metrics for each model
    results_dict = {}
    for model in model_list:
        print(f"\n{'='*60}")
        print(f"Processing model: {model}")
        print(f"{'='*60}")
        results = get_model_metrics(model, model_path_mapping)
        # Always add results to dict (even if metrics are blank)
        results_dict[model] = results
    
    # Write results to file
    write_metrics_file(model_list, results_dict, output_path)

if __name__ == "__main__":
    main()

