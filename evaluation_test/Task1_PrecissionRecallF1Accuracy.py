import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    # print("*******")
    # print(y_true.to_string())
    # print(y_pred.to_string())


    
    if feature_name != 'verbal_responsiveness':
    # Filter out any non-standard values
        valid_values = ['yes', 'no']  # All features use binary yes/no values
        valid_mask = y_true.isin(valid_values) & y_pred.isin(valid_values)
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
    else:
        y_pred = y_pred.fillna('na')
        y_pred = y_pred.replace('NaN', 'na')  
        valid_values = ['yes', 'no','na']  # All features use binary yes/no values
        valid_mask = y_true.isin(valid_values) & y_pred.isin(valid_values)
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]  
    
    if len(y_true) == 0 or len(y_pred) == 0:
        return '', '', '', ''
    
    if feature_name == 'verbal_responsiveness':
        # print(y_true.to_string())
        # print(y_pred.to_string())
        # Multi-class classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    else:
        # Binary classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label='yes', zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label='yes', zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label='yes', zero_division=0)
    
    return accuracy, precision, recall, f1

def get_model_metrics(model_name, features, model_path_mapping):
    """Calculate metrics for a specific model
    Args:
        model_name (str): Name of the model to evaluate
        features (list): List of features to evaluate
        model_path_mapping (dict): Mapping from display model names to actual file path names
    Returns:
        dict: Dictionary containing metrics for each feature
    """
    # File paths
    gt_path = 'result/ground_truth/task12_annotation.csv'
    
    # Get actual path name from mapping, or use model_name if not in mapping
    actual_model_name = model_path_mapping.get(model_name, model_name)
    
    # Special case for audio-flamingo-3
    if model_name.lower() == 'audio-flamingo-3':
        pred_path = f'result/vlm_inference_test/{actual_model_name}/Task12_AF3_features_all.csv'
    else:
        pred_path = f'result/vlm_inference_test/{actual_model_name}/Task12_{actual_model_name}_all_merged.csv'
    
    try:
        # Read ground truth and prediction files
        df_gt = pd.read_csv(gt_path, encoding='latin-1')
        df_pred = pd.read_csv(pred_path, encoding='latin-1')
        
        # Handle file name differences for audio-flamingo-3
        if model_name.lower() == 'audio-flamingo-3':
            # Convert .mp4 to .wav in ground truth file names
            df_gt['file_name'] = df_gt['file_name'].str.replace('.mp4', '.wav')
        
        # Merge ground truth and predictions
        merged_df = pd.merge(df_gt, df_pred, on='file_name', how='inner', 
                           suffixes=('', '_pred'))
        
        # Initialize results dictionary
        results = {}
        
        # Calculate metrics for each feature
        for feature in features:
            try:
                # Get ground truth and prediction columns
                y_true = merged_df[feature]
                y_pred = merged_df[f'{feature}_pred']
                
                # Special handling for close_eyes and eye_blinking
                if feature in ['close_eyes', 'eye_blinking']:
                    valid_indices = y_true != 'nan'  # Filter out NA samples
                    y_true = y_true[valid_indices]
                    y_pred = y_pred[valid_indices]
                
                # Calculate metrics
                try:
                    accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred, feature)
                    
                    # Store results
                    results[feature] = {
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'accuracy': accuracy
                    }
                except Exception as e:
                    print(f"Error calculating metrics for feature {feature}: {str(e)}")
                    results[feature] = {
                        'precision': '',
                        'recall': '',
                        'f1': '',
                        'accuracy': ''
                    }
                
                print(f"Processed feature: {feature}")
                print(f"Metrics - Precision: {float(precision):.2f}, Recall: {float(recall):.2f}, F1: {float(f1):.2f}, Accuracy: {float(accuracy):.2f}")
                
            except KeyError as e:
                print(f"Warning: Feature {feature} not found in data. Error: {e}")
                results[feature] = {
                    'precision': '',
                    'recall': '',
                    'f1': '',
                    'accuracy': ''
                }
        
        return results
    
    except Exception as e:
        print(f"Error processing files for model {model_name}: {str(e)}")
        return None

def write_metrics_file(model_names, features, results_dict, output_path):
    """Write metrics to CSV file
    Args:
        model_names (list): List of model names
        features (list): List of features
        results_dict (dict): Dictionary containing metrics for each model
        output_path (str): Path to output file
    """
    try:
        with open(output_path, 'w') as f:
            # Create header row with feature_name_metric_name format
            metric_names = ['precision', 'recall', 'f1', 'accuracy']
            header = ['model']
            for feature in features:
                for metric in metric_names:
                    header.append(f"{feature}_{metric}")
            f.write(','.join(header) + '\n')
            
            # Write data rows
            for model in model_names:
                row = [model]
                for feature in features:
                    if model in results_dict and results_dict[model] and feature in results_dict[model]:
                        metrics = results_dict[model][feature]
                        row.extend([
                            f"{float(metrics['precision']):.2f}" if metrics['precision'] != '' else '',
                            f"{float(metrics['recall']):.2f}" if metrics['recall'] != '' else '',
                            f"{float(metrics['f1']):.2f}" if metrics['f1'] != '' else '',
                            f"{float(metrics['accuracy']):.2f}" if metrics['accuracy'] != '' else ''
                        ])
                    else:
                        row.extend(['', '', '', ''])
                f.write(','.join(row) + '\n')
        
        print(f"Metrics saved to {output_path}")
    
    except Exception as e:
        print(f"Error writing metrics file: {str(e)}")
        raise

def main():
    # List of features to evaluate
    features = [
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
    
    # Model names
    # model_names = [
    #     #'Qwen2.5-VL-7B',
    #     # 'InternVL3.5-8B',
    #     # 'Qwen2.5-VL-32B',
    #     # 'InternVL3.5-38B',
    #     # 'Qwen2.5-VL-72B',
    #     'Audio-flamingo-3',
    #     # 'Qwen2.5-Omni-7B',
    #     # 'Lingshu-32B',
    #     # 'Qwen3-VL-8B',
    #     # 'Qwen3-VL-32B'
    # ]


    MODELS = [
        "InternVL3_5-8B",
        "Qwen2.5-VL-7B-Instruct",
        'Qwen3-VL-8B-Instruct',
        #"InternVL3_5-38B",
        "Qwen2.5-VL-32B-Instruct",
        'Qwen3-VL-32B-Instruct',
        "Qwen2.5-VL-72B-Instruct",
        'audio-flamingo-3',
        "Qwen2.5-Omni-7B",
        "Qwen3-Omni-30B-A3B-Instruct",
        "seizure_omni_sft"
        #"Lingshu-32B",
    ]  
    
    # Mapping from display model names to actual file path names
    model_path_mapping = {}
    #     'Qwen2.5-VL-7B': 'Qwen2.5-VL-7B-Instruct',
    #     'InternVL3.5-8B': 'InternVL3_5-8B',
    #     'Qwen2.5-VL-32B': 'Qwen2.5-VL-32B-Instruct',
    #     'InternVL3.5-38B': 'InternVL3_5-38B',
    #     'Qwen2.5-VL-72B': 'Qwen2.5-VL-72B-Instruct',
    #     'Audio-flamingo-3': 'audio-flamingo-3',
    #     'Qwen3-VL-8B': 'Qwen3-VL-8B-Instruct',
    #     'Qwen3-VL-32B': 'Qwen3-VL-32B-Instruct'
    # }
    
    output_path = 'metrics_test/Task1_precision_recall_f1_accuracy.csv'
    
    # Calculate metrics for each model
    results_dict = {}
    for model in MODELS:
        print(f"\nProcessing model: {model}")
        results_dict[model] = get_model_metrics(model, features, model_path_mapping)
    
    # Write results to file
    write_metrics_file(MODELS, features, results_dict, output_path)

if __name__ == "__main__":
    main()