import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

experiment = 'internvl3_78_segment'


df_gt = pd.read_csv('/home/lina/gemini/FeatureAnnotation_V3.csv')
df_pred = pd.read_csv('/home/lina/internvl3/merged_tonmoy_tremor_output_ineternvl2_sleep_newfeature_hands_eyes.csv')

# 数据预处理
df_gt['file_name'] = df_gt['file_name'].str.replace('.m2t', '.mp4').str.strip()#.str.lower()
#df_pred['file_name'] = df_pred['file_name'].str.replace('_segment_1', '').str.strip().str.lower()


# 对齐数据
merged_df = pd.merge(df_gt, df_pred, on='file_name', suffixes=('_gt', '_pred'), how='inner')

# 打印不匹配的file_name
unmatched_gt = df_gt[~df_gt['file_name'].isin(merged_df['file_name'])]
unmatched_pred = df_pred[~df_pred['file_name'].isin(merged_df['file_name'])]
print("Unmatched file_name in ground truth:")
print(unmatched_gt['file_name'])
print("Unmatched file_name in predictions:")
print(unmatched_pred['file_name'])

# 定义计算指标的函数
def calculate_metrics(y_true, y_pred, positive_label):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    return accuracy, precision, recall, f1

# 初始化结果字典
results = {'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Positive Num']}

def map_responsiveness(value):
    if value == 'responsive':
        return 'yes'
    elif value == 'non-responsive':
        return 'no'
    else:
        return 'na'



# 遍历每一列特征
for col in df_pred.columns[1:]:
    print(col)
    #if col in ['full_body_jerking' , 'verbal_responsiveness', 'start_time','end_time','label','segment_num','tonic_clonic','motor_pattern','limb_movements_pattern']:
    print(col not in  df_gt.columns[1:])
    print(col  in ['label'])
    if (col not in  df_gt.columns[1:]) or (col  in ['label','verbal_responsiveness']):
        continue
    y_true = merged_df[col + '_gt'].astype(str).str.replace('.', '', regex=False).str.strip().str.lower()
    y_pred = merged_df[col + '_pred'].astype(str).str.replace('.', '', regex=False).str.strip().str.lower()
    
    

    # 对close_eyes列特殊处理：去掉ground truth为NA的样本
    if col == 'close_eyes' or col == 'eye_blinking':
        valid_indices = y_true != 'nan'  # 过滤掉ground truth为NA的样本
        y_true = y_true[valid_indices]
        y_pred = y_pred[valid_indices]

    # if col == 'verbal_responsiveness':
    #     valid_indices = (y_true != 'na' and y_pred != 'na' ) # 过滤掉ground truth为NA的样本
    #     y_true = y_true[valid_indices]
    #     y_pred = y_pred[valid_indices]    
    
    # 确定正样本标签
    positive_label = 'female' if col == 'gender' else 'yes'
    
    # 计算指标
    accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred, positive_label)

    positive_num = (y_true == positive_label).sum()
    
    # 将结果添加到字典中
    print([accuracy, precision, recall, f1, positive_num])
    results[col] = [accuracy, precision, recall, f1, positive_num]

# 将结果保存到CSV文件
results_df = pd.DataFrame(results)
results_df = results_df.round(2)
results_df.to_csv(experiment + '_metrics.csv', index=False)

