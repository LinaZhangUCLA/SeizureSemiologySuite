import numpy as np

def calculate_metrics(true_sequence, pred_sequence):
    """
    计算两个事件序列之间的多种评估指标
    
    参数:
    true_sequence: 真实事件序列
    pred_sequence: 预测事件序列
    
    返回:
    包含所有指标的字典, 包括序列编辑距离，范
    """
    # 转换为列表格式（如果输入是字符串）
    if isinstance(true_sequence, str):
        true_sequence = [event.strip() for event in true_sequence.split(',')]
    if isinstance(pred_sequence, str):
        pred_sequence = [event.strip() for event in pred_sequence.split(',')]
    
    # 1. 序列编辑距离 (Levenshtein距离)
    edit_distance = calculate_edit_distance(true_sequence, pred_sequence)
    
    # 2. f1, precision, recall
    temporal_metrics = calculate_temporal_metrics(true_sequence, pred_sequence)
    
    # 3. 最长公共子序列 (LCS) 与真实序列长度的比值
    lcs_ratio = calculate_lcs_ratio(true_sequence, pred_sequence)
    
    return {
        'edit_distance': edit_distance,
        'temporal_precision': temporal_metrics['precision'],
        'temporal_recall': temporal_metrics['recall'],
        'temporal_f1': temporal_metrics['f1'],
        'lcs_ratio': lcs_ratio
    }

def calculate_edit_distance(seq1, seq2):
    """
    计算两个序列之间的编辑距离 (Levenshtein距离)
    """
    m, n = len(seq1), len(seq2)
    dp = np.zeros((m+1, n+1))
    
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
        
    for i in range(1, m+1):
        for j in range(1, n+1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j] + 1,  # 删除
                              dp[i][j-1] + 1,  # 插入
                              dp[i-1][j-1] + 1)  # 替换
                
    return dp[m][n]

def calculate_temporal_metrics(true_seq, pred_seq):
    """
    计算时序关系精确率、召回率和F1分数
    
    通过比较事件对之间的时序关系来评估预测序列的质量
    """
    # 提取所有唯一事件
    events = set(true_seq) | set(pred_seq)
    
    # 创建事件到索引的映射
    true_positions = {event: [] for event in events}
    pred_positions = {event: [] for event in events}
    
    # 记录每个事件在序列中的位置
    for i, event in enumerate(true_seq):
        true_positions[event].append(i)
    for i, event in enumerate(pred_seq):
        pred_positions[event].append(i)
    
    true_pairs = set()
    pred_pairs = set()
    
    # 生成真实序列中的时序关系对
    for i in range(len(true_seq)):
        for j in range(i+1, len(true_seq)):
            true_pairs.add((true_seq[i], true_seq[j], j-i))
    
    # 生成预测序列中的时序关系对
    for i in range(len(pred_seq)):
        for j in range(i+1, len(pred_seq)):
            pred_pairs.add((pred_seq[i], pred_seq[j], j-i))
    
    # 计算匹配的时序关系对
    matched_pairs = true_pairs & pred_pairs
    
    # precision, recall, f1
    precision = len(matched_pairs) / len(pred_pairs) if len(pred_pairs) > 0 else 0
    recall = len(matched_pairs) / len(true_pairs) if len(true_pairs) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

def calculate_lcs_ratio(true_seq, pred_seq):
    """
    计算最长公共子序列(LCS)与真实序列长度的比值
    """
    m, n = len(true_seq), len(pred_seq)
    dp = np.zeros((m+1, n+1))
    for i in range(1, m+1):
        for j in range(1, n+1):
            if true_seq[i-1] == pred_seq[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    #LCS长度
    lcs_length = dp[m][n]
    # 计算LCS与真实序列长度的比值
    lcs_ratio = lcs_length / m if m > 0 else 0
    
    return lcs_ratio


if __name__ == "__main__":

    true_seq = ["blank_stare", "close_eyes", "eye_blinking", "face_twitching"]
    pred_seq = ["blank_stare", "close_eyes", "head_turning", "eye_blinking"]
    
    metrics = calculate_metrics(true_seq, pred_seq)
    
    print("序列编辑距离:", metrics['edit_distance'])
    print("时序关系精确率:", metrics['temporal_precision'])
    print("时序关系召回率:", metrics['temporal_recall'])
    print("时序关系F1分数:", metrics['temporal_f1'])
    print("LCS与真实序列长度比值:", metrics['lcs_ratio'])
