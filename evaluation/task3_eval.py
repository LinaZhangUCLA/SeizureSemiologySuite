

def calculate_metrics(input_csv, ground_truth):
    import pandas as pd
    df = pd.read_csv(input_csv)
    ground_truth_csv = pd.read_csv(ground_truth)
    ground_truth_csv['file_name'] = ground_truth_csv['file_name'].str[:-4]
    
    # 提取前缀和segment编号
    df['prefix'] = df['file_name'].apply(lambda x: x.split('_segment_')[0])
    df['segment_num'] = df['file_name'].apply(lambda x: int(x.split('_segment_')[1].split('.')[0]))
    
    # 定义合并序列的函数
    def merge_sequences(sequences):
        if not sequences:
            return []
        merged = sequences[0]
        for i in range(1, len(sequences)):
            next_seq = sequences[i]
            k_prev = min(3, len(merged))
            k_next = min(3, len(next_seq))
            overlap_prev = merged[-k_prev:]
            overlap_next = next_seq[:k_next]
            if overlap_prev == overlap_next:
                merged.extend(next_seq[k_next:])
            else:
                merged.extend(next_seq)
        return merged

    
    
    # 按前缀分组，排序并合并序列
    results = []
 
    for prefix, group in df.groupby('prefix'):
        group_sorted = group.sort_values('segment_num')
        sequences = group_sorted['event_sequence'].apply(
            lambda s: [event.strip() for event in s.split(',')] if isinstance(s, str) else []
        ).tolist()
        merged_list = merge_sequences(sequences)
        merged_str = ', '.join(merged_list)
        results.append({'prefix': prefix, 'merged_event_sequence': merged_str})
    
    # 创建新的DataFrame
    result_df = pd.DataFrame(results)
    
     # 方法2：直接在合并时处理
    merged_df = pd.merge(
        result_df,
        ground_truth_csv,
        left_on=result_df['prefix'],
        right_on=ground_truth_csv['file_name'],
        how='inner'
    )
    
    
    import numpy as np
    import Levenshtein
    from seqeval.scheme import Token
    import re
    from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
    from rapidfuzz.distance import LCSseq
    
    def normalize_sequence(seq):
        if isinstance(seq, str):
            seq = re.split(r'[,，、\s]+', seq)
        normalized = [str(item).strip().lower() for item in seq if str(item).strip()]
        return normalized
    
    
    
    
    def calculate_temporal_metrics(true_seq, pred_seq, preprocess=True):
        """
        Calculates precision, recall, and F1-score based on the Longest Common Subsequence (LCS).
        This method respects the relative order of elements in the sequences.
        """
        true_seq_norm = normalize_sequence(true_seq)
        pred_seq_norm = normalize_sequence(pred_seq)
        
        if not true_seq_norm and not pred_seq_norm:
            return 1.0, 1.0, 1.0
        if not true_seq_norm or not pred_seq_norm:
            return 0.0, 0.0, 0.0
    
        if preprocess:
            true_positives = LCSseq.similarity(true_seq_norm, pred_seq_norm)
            precision = true_positives / len(pred_seq_norm)
            recall = true_positives / len(true_seq_norm)
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
        else: 
            true_len = len(true_seq_norm)
            pred_len = len(pred_seq_norm)
            
            if true_len > pred_len:
                # true更长，用'O'标签填充pred
                pred_seq_norm = pred_seq_norm + ['O'] * (true_len - pred_len)
            elif pred_len > true_len:
                # pred更长，截断pred
                pred_seq_norm = pred_seq_norm[:true_len]
        
            precision = precision_score([true_seq_norm], [pred_seq_norm])
            recall = recall_score([true_seq_norm], [pred_seq_norm])
            f1 = f1_score([true_seq_norm], [pred_seq_norm])
    
        return precision, recall, f1
    
    def calculate_metrics(true_sequence, pred_sequence):
        """
        计算两个事件序列之间的多种评估指标
        
        参数:
        true_sequence: 真实事件序列 (列表)
        pred_sequence: 预测事件序列 (列表)
        
        返回:
        包含所有指标的字典
        """
        if isinstance(true_sequence, str):
            true_sequence = [event.strip() for event in true_sequence.split(',')]
        if isinstance(pred_sequence, str):
            pred_sequence = [event.strip() for event in pred_sequence.split(',')]
        
        # 1. 序列编辑距离 (Levenshtein距离)
        edit_distance = calculate_edit_distance(true_sequence, pred_sequence)
        
        precision, recall, f1 = calculate_temporal_metrics(true_sequence, pred_sequence)
        
        # 3. 最长公共子序列 (LCS) 与真实序列长度的比值
        lcs_ratio = calculate_lcs_ratio(true_sequence, pred_sequence)
        
        return {
            'edit_distance': edit_distance,
            'temporal_precision': precision,
            'temporal_recall': recall,
            'temporal_f1': f1,
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
     
    
    def calculate_lcs_ratio(true_seq, pred_seq):
        def normalize_string(s):
            try:
                # 直接尝试转换为字符串并处理
                return re.sub(r'[^\w]', '', str(s)).lower()
            except:
                # 任何异常都返回空字符串
                return ""
        lcs_ratio = LCSseq.normalized_similarity(true_seq, pred_seq, processor=normalize_string)
        return lcs_ratio
    
    
    
    
    ans_list = []
    for i in range(len(merged_df)):
        events = merged_df['merged_event_sequence'][i]
        pred = [event.strip() for event in events.split(',')]
        
        events = merged_df['event_sequence'][i]g
        ground = [event.strip() for event in events.split(',')]
        ans = calculate_metrics(ground, pred)
        ans_list.append(ans)
    
    # 提取所有值并计算平均值
    avg_edit_distance = np.mean([x['edit_distance'] for x in ans_list])
    avg_temporal_precision = np.mean([x['temporal_precision'] for x in ans_list])
    avg_temporal_recall = np.mean([x['temporal_recall'] for x in ans_list])
    avg_temporal_f1 = np.mean([x['temporal_f1'] for x in ans_list])
    avg_lcs_ratio = np.mean([x['lcs_ratio'] for x in ans_list])
    
    
    
    print(f"编辑距离: {avg_edit_distance}")
    print(f"精度: {avg_temporal_precision}")
    print(f"召回率: {avg_temporal_recall}")
    print(f"F1分数: {avg_temporal_f1}")
    print(f"LCS比率: {avg_lcs_ratio}")
    
    metrics = {
        'Sequence Edit Distance': avg_edit_distance,
        'Temporal Relation P': avg_temporal_precision,
        'Temporal Relation R': avg_temporal_recall,
        'Temporal Relation F1': avg_temporal_f1,
        'LCS Ratio': avg_lcs_ratio
    }
    return metrics 
