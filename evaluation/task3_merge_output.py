results_csv = './Qwen2.5-VL-7B-Instruct/Task3_Qwen2.5-VL-7B-Instruct_all.csv'
storage_csv = 'Qwen2.5-VL-7B-Instruct/Task3_Qwen2.5-VL-7B-Instruct_all_merge.csv'

df = pd.read_csv(results_csv)

# 提取前缀和segment编号
df['prefix'] = df['file_name'].apply(lambda x: x.split('_segment_')[0])
df['segment_num'] = df['file_name'].apply(lambda x: int(x.split('_segment_')[1].split('.')[0]))

# 定义合并序列的函数, 后三个和前三个match。如果不match。那么就直接拼接
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
    sequences = group_sorted['event_sequence'].apply(lambda s: [event.strip() for event in s.split(',')]).tolist()
    merged_list = merge_sequences(sequences)
    merged_str = ', '.join(merged_list)
    results.append({'prefix': prefix, 'merged_event_sequence': merged_str})


result_df = pd.DataFrame(results) 
print(result_df.head(3))
result_df.to_csv(storage_csv)
