import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def calculate_metrics(y_true, y_pred, feature_name=None):
    """Calculate metrics for classification
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        feature_name: Name of the feature (for special handling)
    Returns:
        tuple: (accuracy, precision, recall, f1)
    """
    # Convert to lowercase and strip whitespace
    y_true = y_true.str.strip().str.lower()
    y_pred = y_pred.str.strip().str.lower()
    
    if feature_name == 'body_region_onset':
        # Multi-class classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    else:
        # Binary classification metrics (left/right)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label='right', zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label='right', zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label='right', zero_division=0)
    
    return accuracy, precision, recall, f1

def get_model_metrics(model_name, model_path_name):
    """Calculate metrics for a specific model
    Args:
        model_name (str): Display name of the model
        model_path_name (str): Actual file path name of the model
    Returns:
        dict: Dictionary containing metrics for each feature
    """
    # File paths
    gt_path = 'result/ground_truth/task3_spatial_annotation.csv'
    
    try:
        # Read ground truth file
        df_gt = pd.read_csv(gt_path, encoding='latin-1')
        
        # Initialize results dictionary
        results = {}
        
        # Process head turning feature
        try:
            pred_path = f'result/vlm_inference/{model_path_name}/Task3_HT_{model_path_name}_1-129.csv'
            df_pred = pd.read_csv(pred_path, encoding='latin-1')
            
            # Merge ground truth and predictions
            # Handle both video_name and file_name columns
            merge_col = 'video_name' if 'video_name' in df_pred.columns else 'file_name'
            merged_df = pd.merge(df_gt, df_pred, left_on='file_name', right_on=merge_col, how='inner')
            
            # Filter out invalid values
            valid_mask = (merged_df['left_right_head_turning'].isin(['left', 'right'])) & \
                        (merged_df['head_turning_direction'].isin(['left', 'right']))
            
            if valid_mask.any():
                accuracy, precision, recall, f1 = calculate_metrics(
                    merged_df[valid_mask]['left_right_head_turning'],
                    merged_df[valid_mask]['head_turning_direction']
                )
                results['left_right_head_turning'] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': accuracy
                }
            else:
                results['left_right_head_turning'] = {
                    'precision': '',
                    'recall': '',
                    'f1': '',
                    'accuracy': ''
                }
        except Exception as e:
            print(f"Error processing head turning for model {model_name}: {str(e)}")
            results['left_right_head_turning'] = {
                'precision': '',
                'recall': '',
                'f1': '',
                'accuracy': ''
            }
            
        # Process arm movement feature
        try:
            pred_path = f'result/vlm_inference/{model_path_name}/Task3_AM_{model_path_name}_1-112.csv'
            df_pred = pd.read_csv(pred_path, encoding='latin-1')
            
            # Merge ground truth and predictions
            # Handle both video_name and file_name columns
            merge_col = 'video_name' if 'video_name' in df_pred.columns else 'file_name'
            merged_df = pd.merge(df_gt, df_pred, left_on='file_name', right_on=merge_col, how='inner')
            
            # Filter out invalid values
            valid_mask = (merged_df['left_right_arm_movement'].isin(['left', 'right'])) & \
                        (merged_df['arm_movement_direction'].isin(['left', 'right']))
            
            if valid_mask.any():
                accuracy, precision, recall, f1 = calculate_metrics(
                    merged_df[valid_mask]['left_right_arm_movement'],
                    merged_df[valid_mask]['arm_movement_direction']
                )
                results['left_right_arm_movement'] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': accuracy
                }
            else:
                results['left_right_arm_movement'] = {
                    'precision': '',
                    'recall': '',
                    'f1': '',
                    'accuracy': ''
                }
        except Exception as e:
            print(f"Error processing arm movement for model {model_name}: {str(e)}")
            results['left_right_arm_movement'] = {
                'precision': '',
                'recall': '',
                'f1': '',
                'accuracy': ''
            }
            
        # Process body region onset feature
        try:
            pred_path = f'result/vlm_inference/{model_path_name}/Task3_bodypart_{model_path_name}_all.csv'
            df_pred = pd.read_csv(pred_path, encoding='latin-1')
            
            # Process predictions - take first prediction for each video
            print(f"Original predictions shape: {df_pred.shape}")
            # Filter to keep only segment_0 predictions
            df_pred = df_pred[df_pred['file_name'].str.contains('_segment_0.mp4')]
            # Extract base name by removing segment suffix
            df_pred['base_name'] = df_pred['file_name'].apply(lambda x: x.split('_segment_')[0] + '.mp4')
            print(f"After grouping shape: {df_pred.shape}")
            print("Sample of predictions:")
            print(df_pred.head())
            
            # Merge ground truth and predictions
            merged_df = pd.merge(df_gt, df_pred, left_on='file_name', right_on='base_name', how='inner')
            print(f"After merge shape: {merged_df.shape}")
            print("Sample of merged data:")
            print(merged_df[['base_name', 'body_region_onset', 'onset_body_part']].head())
            
            # Get valid labels from ground truth
            valid_labels = df_gt[df_gt['body_region_onset'].notna()]['body_region_onset'].unique()
            
            # Convert labels to lowercase for comparison
            merged_df['body_region_onset'] = merged_df['body_region_onset'].str.lower()
            merged_df['onset_body_part'] = merged_df['onset_body_part'].str.lower()
            valid_labels = [label.lower() for label in valid_labels]
            
            # Filter out invalid values from both ground truth and predictions
            invalid_values = ['nan', 'NAN', 'filled', 'N/A']
            valid_mask = (merged_df['body_region_onset'].isin(valid_labels)) & \
                        (merged_df['onset_body_part'].notna()) & \
                        (~merged_df['onset_body_part'].isin(invalid_values))
            
            if valid_mask.any():
                accuracy, precision, recall, f1 = calculate_metrics(
                    merged_df[valid_mask]['body_region_onset'],
                    merged_df[valid_mask]['onset_body_part'],
                    feature_name='body_region_onset'
                )
                results['body_region_onset'] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': accuracy
                }
            else:
                results['body_region_onset'] = {
                    'precision': '',
                    'recall': '',
                    'f1': '',
                    'accuracy': ''
                }
        except Exception as e:
            print(f"Error processing body region onset for model {model_name}: {str(e)}")
            results['body_region_onset'] = {
                'precision': '',
                'recall': '',
                'f1': '',
                'accuracy': ''
            }
        
        return results
    
    except Exception as e:
        print(f"Error processing files for model {model_name}: {str(e)}")
        return None

def write_metrics_file(model_names, results_dict, output_path):
    """Write metrics to CSV file
    Args:
        model_names (list): List of model names
        results_dict (dict): Dictionary containing metrics for each model
        output_path (str): Path to output file
    """
    try:
        # Create header row
        features = ['left_right_head_turning', 'left_right_arm_movement', 'body_region_onset']
        metrics = ['precision', 'recall', 'f1', 'accuracy']
        
        header = ['model']
        for feature in features:
            for metric in metrics:
                header.append(f"{feature}_{metric}")
        
        # Write to CSV
        with open(output_path, 'w') as f:
            # Write header
            f.write(','.join(header) + '\n')
            
            # Write data rows
            for model in model_names:
                row = [model]
                if model in results_dict and results_dict[model]:
                    for feature in features:
                        if feature in results_dict[model]:
                            metrics = results_dict[model][feature]
                            row.extend([
                                f"{float(metrics['precision']):.2f}" if metrics['precision'] != '' else '',
                                f"{float(metrics['recall']):.2f}" if metrics['recall'] != '' else '',
                                f"{float(metrics['f1']):.2f}" if metrics['f1'] != '' else '',
                                f"{float(metrics['accuracy']):.2f}" if metrics['accuracy'] != '' else ''
                            ])
                        else:
                            row.extend(['', '', '', ''])
                else:
                    row.extend([''] * len(features) * len(metrics))
                f.write(','.join(row) + '\n')
        
        print(f"Metrics saved to {output_path}")
    
    except Exception as e:
        print(f"Error writing metrics file: {str(e)}")
        raise

def main():
    # Model names
    model_names = [
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
    model_path_mapping = {
        'Qwen2.5-VL-7B': 'Qwen2.5-VL-7B-Instruct',
        'InternVL3.5-8B': 'InternVL3_5-8B',
        'Qwen2.5-VL-32B': 'Qwen2.5-VL-32B-Instruct',
        'InternVL3.5-38B': 'InternVL3_5-38B',
        'Qwen2.5-VL-72B': 'Qwen2.5-VL-72B-Instruct',
        'Audio-flamingo-3': 'audio-flamingo-3',
        'Qwen2.5-Omni-7B': 'Qwen2.5-Omni-7B',
        'Lingshu-32B': 'Lingshu-32B',
        'Qwen3-VL-8B': 'Qwen3-VL-8B-Instruct',
        'Qwen3-VL-32B': 'Qwen3-VL-32B-Instruct'
    }
    
    output_path = 'metrics/Task3_spatial_metrics.csv'
    
    # Calculate metrics for each model
    results_dict = {}
    for model in model_names:
        print(f"\nProcessing model: {model}")
        model_path_name = model_path_mapping.get(model, model)
        results_dict[model] = get_model_metrics(model, model_path_name)
    
    # Write results to file
    write_metrics_file(model_names, results_dict, output_path)

if __name__ == "__main__":
    main()
