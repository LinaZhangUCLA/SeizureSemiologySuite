import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_absolute_error, mean_squared_error
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import editdistance
from typing import List, Tuple, Dict, Any
import re
from bert_score import BERTScorer, score   # has limit of ~ 512 tokens
import nltk
nltk.download('wordnet')
from nltk.translate.meteor_score import meteor_score


# NOTES ON METRICS:
# - BLEU: I dont think BLEU is a good metric for what we are trying to achieve
# - 

def bert_score(reference=None, candidate=None):

    # Initialize the BERTScorer object
    scorer = BERTScorer(lang='en')
    
    # Compute the BERTScore
    P, R, F1 = scorer.score(candidate, reference)

    # Print the scores
    print("Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}".format(P.item(), R.item(), F1.item()))
    return {'precision': P, 'recall': R, 'f1': F1}



# Task 3: Temporal & Sequential Analysis
def kendall_tau(pred_sequence: List, true_sequence: List) -> float:
    tau, _ = kendalltau(pred_sequence, true_sequence)
    return tau

def sequence_edit_distance(pred_sequence: List, true_sequence: List) -> int:
    return editdistance.eval(pred_sequence, true_sequence)

def normalized_edit_distance(pred_sequence: List, true_sequence: List) -> float:
    edit_dist = editdistance.eval(pred_sequence, true_sequence)
    max_len = max(len(pred_sequence), len(true_sequence))
    return edit_dist / max_len if max_len > 0 else 0.0

def temporal_relation_metrics(pred_relations: List[Tuple], true_relations: List[Tuple]) -> Dict[str, float]:
    pred_set = set(pred_relations)
    true_set = set(true_relations)
    
    tp = len(pred_set.intersection(true_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

# Task 4: Spatial & Anatomical Analysis
def spatial_metrics(y_true: np.ndarray, y_pred: np.ndarray, average='weighted') -> Dict[str, float]:
    return {
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred)
    }

# Task 5: Quantitative Analysis
def temporal_iou(pred_intervals: List[Tuple[float, float]], true_intervals: List[Tuple[float, float]]) -> float:
    total_iou = 0.0
    for pred_start, pred_end in pred_intervals:
        best_iou = 0.0
        for true_start, true_end in true_intervals:
            intersection_start = max(pred_start, true_start)
            intersection_end = min(pred_end, true_end)
            intersection = max(0, intersection_end - intersection_start)
            
            union_start = min(pred_start, true_start)
            union_end = max(pred_end, true_end)
            union = union_end - union_start
            
            iou = intersection / union if union > 0 else 0
            best_iou = max(best_iou, iou)
        total_iou += best_iou
    
    return total_iou / len(pred_intervals) if pred_intervals else 0.0

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_absolute_error(y_true, y_pred)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Task 6: Holistic Narrative / Report Generation
def bleu_score(reference: str, candidate: str, n_gram: int = 4) -> float:
    reference_tokens = reference.lower().split()
    candidate_tokens = candidate.lower().split()
    
    smoothie = SmoothingFunction().method4
    if n_gram == 1:
        weights = (1, 0, 0, 0)
    elif n_gram == 2:
        weights = (0.5, 0.5, 0, 0)
    elif n_gram == 3:
        weights = (0.33, 0.33, 0.33, 0)
    else:
        weights = (0.25, 0.25, 0.25, 0.25)
    
    return sentence_bleu([reference_tokens], candidate_tokens, weights=weights, smoothing_function=smoothie)

def rouge_scores(reference: str, candidate: str) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        'rouge1_f1': scores['rouge1'].fmeasure,
        'rouge2_f1': scores['rouge2'].fmeasure,
        'rougeL_f1': scores['rougeL'].fmeasure
    }


def meteor_score_wrapper(reference: str, candidate: str) -> float:
    try:
        reference_tokens = reference.lower().split()
        candidate_tokens = candidate.lower().split()
        return meteor_score([reference_tokens], candidate_tokens)
    except ImportError:
        print("NLTK meteor not available. Install with: pip install nltk")
        return 0.0

def medical_entity_overlap(reference: str, candidate: str, medical_entities: List[str]) -> float:
    ref_entities = set()
    cand_entities = set()
    
    for entity in medical_entities:
        if entity.lower() in reference.lower():
            ref_entities.add(entity)
        if entity.lower() in candidate.lower():
            cand_entities.add(entity)
    
    if not ref_entities:
        return 1.0 if not cand_entities else 0.0
    
    overlap = len(ref_entities.intersection(cand_entities))
    return overlap / len(ref_entities)

def structural_completeness_score(reference: str, candidate: str, required_sections: List[str]) -> float:
    ref_sections = 0
    cand_sections = 0
    matched_sections = 0
    
    for section in required_sections:
        ref_has = section.lower() in reference.lower()
        cand_has = section.lower() in candidate.lower()
        
        if ref_has:
            ref_sections += 1
        if cand_has:
            cand_sections += 1
        if ref_has and cand_has:
            matched_sections += 1
    
    if ref_sections == 0:
        return 1.0 if cand_sections == 0 else 0.0
    
    return matched_sections / ref_sections

# Example usage functions
def evaluate_temporal_analysis(pred_sequence, true_sequence, pred_relations, true_relations):
    results = {}
    results['kendall_tau'] = kendall_tau(pred_sequence, true_sequence)
    results['edit_distance'] = sequence_edit_distance(pred_sequence, true_sequence)
    results['normalized_edit_distance'] = normalized_edit_distance(pred_sequence, true_sequence)
    results.update(temporal_relation_metrics(pred_relations, true_relations))
    return results

def evaluate_spatial_analysis(y_true, y_pred):
    return spatial_metrics(y_true, y_pred)

def evaluate_quantitative_analysis(pred_intervals, true_intervals, y_true_values, y_pred_values):
    results = {}
    results['tiou'] = temporal_iou(pred_intervals, true_intervals)
    results['mae'] = mae(y_true_values, y_pred_values)
    results['rmse'] = rmse(y_true_values, y_pred_values)
    results['mape'] = mape(y_true_values, y_pred_values)
    return results

def evaluate_narrative_generation(reference, candidate, medical_entities=None, required_sections=None):
    results = {}
    results['bleu_4'] = bleu_score(reference, candidate, 4)
    results.update(rouge_scores(reference, candidate))
    # results.update(bertscore_wrapper([reference], [candidate]))
    results['meteor'] = meteor_score_wrapper(reference, candidate)
    
    if medical_entities:
        results['medical_entity_overlap'] = medical_entity_overlap(reference, candidate, medical_entities)
    
    if required_sections:
        results['structural_completeness'] = structural_completeness_score(reference, candidate, required_sections)
    
    return results


def sequential_tests():
    seq1 = [1,2,3,4]
    seq2 = [2,3,5,4]
    
    tau = kendall_tau(seq1,seq2)
    print("Tau (-1,1): ", tau)
    
    

def main():
    
    # Experiment 1: Precision: 0.90, Recall: 0.92, F1: 0.91

    # chat = """BPM (beats per minute) measures the tempo or speed of a song by counting how many beats occur in one minute."""#To find the BPM, you can listen to the song and count the number of beats within a 15-second interval, then multiply that number by four. Musicians and producers often use metronomes or digital audio software to determine or set the exact BPM. Understanding BPM helps match the rhythm of different tracks, especially in dance music, DJing, or composing."""
    # claude = """BPM (beats per minute) measures the tempo or speed of a song by counting how many beats occur within one minute of music."""#To find the BPM, you can tap along to the beat of a song for 15 seconds, count the taps, then multiply by 4 to get the full minute count. Most modern music software and apps can automatically detect BPM by analyzing the audio waveform and identifying the rhythmic pulse patterns. A slow ballad might have a BPM around 60-80, while fast dance music can reach 120-140 BPM or higher.RetryClaude can make mistakes. Please double-check responses."""

    # Experiment 2: Precision: 0.83, Recall: 0.82, F1: 0.83
    chat = "the sky is blue"
    claude = "what is the constitution of the United states of america?"
    
    # Experiment 3: Precision: 0.67, Recall: 0.79, F1: 0.73
    chat = "the sky is blue"
    claude = "894357nc-9578457-98ncnjdslkfgj"
    claude = "the sky is bluer today"

    
    # bert_score(reference=[chat], candidate=[claude])
    # print(bleu_score(reference=chat, candidate=claude))
    print(meteor_score_wrapper(reference=chat, candidate=claude))
    

if __name__ == "__main__":
    main()
    # sequential_tests()