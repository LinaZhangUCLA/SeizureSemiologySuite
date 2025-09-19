import pandas as pd
import numpy as np

df = pd.read_csv('Task1_precision_recall_f1_accuracy.csv')

model_order = [
    'Qwen2.5-VL-7B-Instruct',
    'InternVL3_5-8B', 
    'Qwen2.5-VL-32B-Instruct',
    'InternVL3_5-38B',
    'Qwen2.5-VL-72B-Instruct',
    'Qwen2.5-Omni-7B',
    'audio-flamingo-3',
    'Lingshu-32B'
]

for model in model_order:
    row = df[df['model'] == model].iloc[0]
    print(f"\n{model}:")
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        metric_cols = [col for col in df.columns if col.endswith(f'_{metric}')]
        values = row[metric_cols].dropna().values
        
        if len(values) > 0:
            mean = np.mean(values)
            std = np.std(values)
            print(f"  {metric}: {mean:.2f} ± {std:.2f}")
        else:
            print(f"  {metric}: N/A")
