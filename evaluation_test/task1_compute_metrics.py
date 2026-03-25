import os
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


FEATURES = [
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
    'ictal_vocalization',
]


def calculate_metrics(y_true, y_pred, feature_name=None):
    y_true = y_true.astype(str).str.strip().str.lower()
    y_pred = y_pred.astype(str).str.strip().str.lower()

    if feature_name != 'verbal_responsiveness':
        valid_values = ['yes', 'no']
        valid_mask = y_true.isin(valid_values) & y_pred.isin(valid_values)
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
    else:
        y_pred = y_pred.fillna('na').replace('nan', 'na')
        valid_values = ['yes', 'no', 'na']
        valid_mask = y_true.isin(valid_values) & y_pred.isin(valid_values)
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

    if len(y_true) == 0 or len(y_pred) == 0:
        return None

    if feature_name == 'verbal_responsiveness':
        return {
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
        }

    return {
        'precision': precision_score(y_true, y_pred, pos_label='yes', zero_division=0),
        'recall': recall_score(y_true, y_pred, pos_label='yes', zero_division=0),
        'f1': f1_score(y_true, y_pred, pos_label='yes', zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
    }


def compute_metrics(pred_csv, gt_csv):
    df_gt = pd.read_csv(gt_csv, encoding='latin-1')
    df_pred = pd.read_csv(pred_csv, encoding='latin-1')
    merged_df = pd.merge(df_gt, df_pred, on='file_name', how='inner', suffixes=('', '_pred'))

    results = {}
    for feature in FEATURES:
        pred_col = f'{feature}_pred'
        if feature not in merged_df.columns or pred_col not in merged_df.columns:
            continue

        y_true = merged_df[feature]
        y_pred = merged_df[pred_col]

        if feature in ['close_eyes', 'eye_blinking']:
            valid_indices = y_true.astype(str).str.lower() != 'nan'
            y_true = y_true[valid_indices]
            y_pred = y_pred[valid_indices]

        metrics = calculate_metrics(y_true, y_pred, feature)
        if metrics is not None:
            results[feature] = metrics

    return results


def build_output_row(model_name, results):
    row = {'model': model_name}
    metric_names = ['precision', 'recall', 'f1', 'accuracy']
    for feature in FEATURES:
        feature_metrics = results.get(feature)
        for metric_name in metric_names:
            key = f'{feature}_{metric_name}'
            row[key] = round(feature_metrics[metric_name], 2) if feature_metrics else ''

    populated = [m for m in results.values() if m]
    if populated:
        row['mean_precision'] = round(sum(m['precision'] for m in populated) / len(populated), 2)
        row['mean_recall'] = round(sum(m['recall'] for m in populated) / len(populated), 2)
        row['mean_f1'] = round(sum(m['f1'] for m in populated) / len(populated), 2)
        row['mean_accuracy'] = round(sum(m['accuracy'] for m in populated) / len(populated), 2)
    else:
        row['mean_precision'] = ''
        row['mean_recall'] = ''
        row['mean_f1'] = ''
        row['mean_accuracy'] = ''
    return row


def parse_args():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Compute Task 1 classification metrics for one merged prediction CSV.")
    parser.add_argument("--pred_csv", required=True, help="Merged Task12 prediction CSV.")
    parser.add_argument("--gt_csv", default=os.path.join(repo_root, "result", "ground_truth", "task12_annotation.csv"),
                        help="Ground-truth Task 1/2 annotation CSV.")
    parser.add_argument("--model_name", default=None, help="Model name to store in the output row.")
    parser.add_argument("--out_csv", required=True, help="Output CSV path.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name or os.path.basename(os.path.dirname(args.pred_csv))
    results = compute_metrics(args.pred_csv, args.gt_csv)
    out_df = pd.DataFrame([build_output_row(model_name, results)])
    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Metrics saved to {args.out_csv}")


if __name__ == "__main__":
    main()
