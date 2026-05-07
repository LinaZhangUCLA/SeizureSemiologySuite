"""
SeizureRQI Scorer - Compare VLM inference results with Ground Truth
This script calculates SeizureRQI scores by comparing VLM-generated analysis 
with ground truth annotations based on multiple criteria.
"""

import pandas as pd
import json
import argparse
from typing import Dict, List
from tqdm import tqdm


class SeizureRQIScorer:
    
    def __init__(self):
        """Initialize the scorer with scoring weights"""
        # Each component worth 100 points, total = 1.0
        self.structural_weight = 0.25      # Structural completeness: 25%
        self.coverage_weight = 0.25        # Symptom coverage: 25%
        self.temporal_f1_weight = 0.25     # Temporal F1: 25%
        self.localizing_weight = 0.25      # Localizing features: 25%
    
    def parse_json_field(self, field_value):
        """Parse JSON string field into Python object"""
        if pd.isna(field_value):
            return None
        if isinstance(field_value, str):
            try:
                return json.loads(field_value)
            except:
                return None
        return field_value
    
    def calculate_structural_completeness(self, vlm_struct, gt_struct):
        """
        Calculate structural completeness score (0-100)
        Score = (VLM components present / GT components present) * 100
        Capped at 100
        """
        if not gt_struct or not vlm_struct:
            return 0.0
        
        # Count components present in GT
        gt_components = [comp for comp in gt_struct if comp in ['onset', 'propagation', 'postictal']]
        if len(gt_components) == 0:
            return 0.0
        
        # Count how many GT components are also in VLM
        vlm_components = [comp for comp in vlm_struct if comp in ['onset', 'propagation', 'postictal']]
        matched = len(set(gt_components) & set(vlm_components))
        
        score = (matched / len(gt_components)) * 100
        return min(100.0, score)
    
    def calculate_coverage(self, vlm_sequence, gt_sequence):
        """
        Calculate feature coverage score (0-100)
        Score = (Number of GT features found in VLM / Total GT features) * 100
        """
        if not gt_sequence:
            return 0.0
        
        gt_features = set(gt_sequence)
        vlm_features = set(vlm_sequence) if vlm_sequence else set()
        
        if len(gt_features) == 0:
            return 0.0
        
        matched_features = len(gt_features & vlm_features)
        score = (matched_features / len(gt_features)) * 100
        return score
    
    def calculate_f1_score(self, vlm_sequence, gt_sequence):
        """
        Calculate F1 score for temporal sequence alignment (0-100)
        Simple token-level F1 calculation
        """
        if not gt_sequence and not vlm_sequence:
            return 100.0
        
        if not gt_sequence or not vlm_sequence:
            return 0.0
        
        # Convert to sets for precision/recall calculation
        vlm_set = set(vlm_sequence)
        gt_set = set(gt_sequence)
        
        # Calculate precision and recall
        true_positives = len(vlm_set & gt_set)
        
        precision = true_positives / len(vlm_set) if len(vlm_set) > 0 else 0
        recall = true_positives / len(gt_set) if len(gt_set) > 0 else 0
        
        # Calculate F1
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1 * 100
    
    def calculate_localizing_features(self, vlm_localizing, gt_localizing):
        """
        Calculate localizing features score (0-100)
        Score = (Number of matched values / Total GT non-none values) * 100
        Only count features where GT has a specific value (not "none")
        """
        if not gt_localizing or not vlm_localizing:
            return 0.0
        
        keys = ['head_turning', 'first_motor_laterality', 'asymmetric_posturing', 'figure_4']
        
        # Count GT features with actual values (not "none")
        gt_valid_count = 0
        matched_count = 0
        
        for key in keys:
            gt_value = gt_localizing.get(key, "none")
            if gt_value and gt_value != "none":
                gt_valid_count += 1
                vlm_value = vlm_localizing.get(key, "none")
                if vlm_value == gt_value:
                    matched_count += 1
        
        if gt_valid_count == 0:
            return 0.0
        
        score = (matched_count / gt_valid_count) * 100
        return score
    
    def calculate_hallucination_penalty(self, vlm_sequence, gt_sequence):
        """
        Calculate hallucination penalty
        -10 points per feature in VLM that's not in GT
        """
        if not vlm_sequence or not gt_sequence:
            return 0
        
        vlm_features = set(vlm_sequence)
        gt_features = set(gt_sequence)
        
        hallucinated = vlm_features - gt_features
        return len(hallucinated)
    
    def calculate_content_penalty(self, vlm_content):
        """
        Calculate content analysis penalty
        - Safety issues: cap score at 50
        - Non-seizure content: -20% penalty
        """
        if not vlm_content:
            return 1.0, False
        
        has_safety = vlm_content.get('has_safety_issues', False)
        has_non_seizure = vlm_content.get('includes_non_seizure', False)
        
        # Non-seizure penalty
        penalty_multiplier = 0.8 if has_non_seizure else 1.0
        
        return penalty_multiplier, has_safety
    
    def calculate_length_penalty(self, vlm_length, gt_length):
        """
        Calculate length penalty
        -1 point per 50 characters over GT length
        """
        if vlm_length <= gt_length:
            return 0
        
        over_length = vlm_length - gt_length
        penalty_points = (over_length // 50)
        return penalty_points
    
    def score_single_report(self, vlm_row, gt_row):
        """
        Score a single report comparison
        Returns a dictionary with all scores and final SeizureRQI
        """
        # Parse JSON fields
        vlm_struct = self.parse_json_field(vlm_row.get('structural_completeness'))
        gt_struct = self.parse_json_field(gt_row.get('structural_completeness'))
        
        vlm_sequence = self.parse_json_field(vlm_row.get('event_sequence'))
        gt_sequence = self.parse_json_field(gt_row.get('event_sequence'))
        
        vlm_localizing = self.parse_json_field(vlm_row.get('localizing_features'))
        gt_localizing = self.parse_json_field(gt_row.get('localizing_features'))
        
        vlm_content = self.parse_json_field(vlm_row.get('content_analysis'))
        
        vlm_length = vlm_row.get('report_length', 0)
        gt_length = gt_row.get('report_length', 0)
        
        # Calculate component scores (each 0-100)
        structural_score = self.calculate_structural_completeness(vlm_struct, gt_struct)
        coverage_score = self.calculate_coverage(vlm_sequence, gt_sequence)
        f1_score = self.calculate_f1_score(vlm_sequence, gt_sequence)
        localizing_score = self.calculate_localizing_features(vlm_localizing, gt_localizing)
        
        # Calculate base weighted score (0-100)
        base_score = (
            self.structural_weight * structural_score +
            self.coverage_weight * coverage_score +
            self.temporal_f1_weight * f1_score +
            self.localizing_weight * localizing_score
        )
        
        # Calculate penalties
        hallucination_count = self.calculate_hallucination_penalty(vlm_sequence, gt_sequence)
        hallucination_penalty = hallucination_count * 1  # 1 point per hallucination
        
        content_multiplier, has_safety = self.calculate_content_penalty(vlm_content)
        
        length_penalty = self.calculate_length_penalty(vlm_length, gt_length)
        
        # Apply penalties
        score_after_hallucination = base_score - hallucination_penalty
        score_after_content = score_after_hallucination * content_multiplier
        final_score = score_after_content - length_penalty
        
        # Safety gate: cap at 50 if safety issues
        if has_safety:
            final_score = min(final_score, 50.0)
        
        # Ensure score is between 0-100
        final_score = max(0.0, min(100.0, final_score))
        
        return {
            'structural_completeness_score': structural_score,
            'coverage_score': coverage_score,
            'temporal_f1_score': f1_score,
            'localizing_features_score': localizing_score,
            'base_score': base_score,
            'hallucination_count': hallucination_count,
            'hallucination_penalty': hallucination_penalty,
            'has_safety_issues': has_safety,
            'has_non_seizure_content': vlm_content.get('includes_non_seizure', False) if vlm_content else False,
            'content_penalty_multiplier': content_multiplier,
            'length_penalty': length_penalty,
            'seizure_rqi': final_score
        }
    
    def score_dataset(self, vlm_csv_path: str, gt_csv_path: str, output_path: str = None):
        """
        Score entire dataset by comparing VLM results with Ground Truth
        """
        # Load data
        print("Loading datasets...")
        vlm_df = pd.read_csv(vlm_csv_path)
        gt_df = pd.read_csv(gt_csv_path)
        
        # Merge on file_name
        print("Merging datasets on file_name...")
        merged_df = pd.merge(
            vlm_df, gt_df,
            on='file_name',
            suffixes=('_vlm', '_gt')
        )
        
        print(f"Found {len(merged_df)} matching reports")
        
        # Score each report
        results = []
        print("Scoring reports...")
        for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
            vlm_data = {
                'structural_completeness': row.get('structural_completeness_vlm'),
                'event_sequence': row.get('event_sequence_vlm'),
                'localizing_features': row.get('localizing_features_vlm'),
                'content_analysis': row.get('content_analysis_vlm'),
                'report_length': row.get('report_length_vlm', 0)
            }
            
            gt_data = {
                'structural_completeness': row.get('structural_completeness_gt'),
                'event_sequence': row.get('event_sequence_gt'),
                'localizing_features': row.get('localizing_features_gt'),
                'report_length': row.get('report_length_gt', 0)
            }
            
            scores = self.score_single_report(vlm_data, gt_data)
            scores['file_name'] = row['file_name']
            results.append(scores)
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Reorder columns
        cols = ['file_name', 'seizure_rqi', 'structural_completeness_score', 
                'coverage_score', 'temporal_f1_score', 'localizing_features_score',
                'base_score', 'hallucination_count', 'hallucination_penalty',
                'has_safety_issues', 'has_non_seizure_content', 
                'content_penalty_multiplier', 'length_penalty']
        results_df = results_df[cols]
        
        # Calculate summary statistics
        print("\n=== Scoring Summary ===")
        print(f"Mean SeizureRQI: {results_df['seizure_rqi'].mean():.2f}")
        print(f"Std SeizureRQI: {results_df['seizure_rqi'].std():.2f}")
        print(f"Min SeizureRQI: {results_df['seizure_rqi'].min():.2f}")
        print(f"Max SeizureRQI: {results_df['seizure_rqi'].max():.2f}")
        print(f"\nMean Structural Completeness: {results_df['structural_completeness_score'].mean():.2f}")
        print(f"Mean Coverage: {results_df['coverage_score'].mean():.2f}")
        print(f"Mean Temporal F1: {results_df['temporal_f1_score'].mean():.2f}")
        print(f"Mean Localizing Features: {results_df['localizing_features_score'].mean():.2f}")
        print(f"\nTotal Hallucinations: {results_df['hallucination_count'].sum()}")
        print(f"Reports with Safety Issues: {results_df['has_safety_issues'].sum()}")
        print(f"Reports with Non-Seizure Content: {results_df['has_non_seizure_content'].sum()}")
        
        # Save results
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")
            
            # Save summary
            summary_path = output_path.replace('.csv', '_summary.json')
            summary = {
                'mean_seizure_rqi': float(results_df['seizure_rqi'].mean()),
                'std_seizure_rqi': float(results_df['seizure_rqi'].std()),
                'min_seizure_rqi': float(results_df['seizure_rqi'].min()),
                'max_seizure_rqi': float(results_df['seizure_rqi'].max()),
                'mean_structural': float(results_df['structural_completeness_score'].mean()),
                'mean_coverage': float(results_df['coverage_score'].mean()),
                'mean_temporal_f1': float(results_df['temporal_f1_score'].mean()),
                'mean_localizing': float(results_df['localizing_features_score'].mean()),
                'total_hallucinations': int(results_df['hallucination_count'].sum()),
                'reports_with_safety_issues': int(results_df['has_safety_issues'].sum()),
                'reports_with_non_seizure': int(results_df['has_non_seizure_content'].sum()),
                'total_reports': len(results_df)
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Summary saved to: {summary_path}")
        
        return results_df


def main():
    parser = argparse.ArgumentParser(description='Calculate SeizureRQI scores by comparing VLM results with Ground Truth')
    parser.add_argument('--vlm_csv', type=str, required=True,
                       help='Path to VLM inference results CSV')
    parser.add_argument('--gt_csv', type=str, required=True,
                       help='Path to Ground Truth analysis CSV')
    parser.add_argument('--output', type=str, default='seizure_rqi_scores.csv',
                       help='Output path for scores CSV')
    
    args = parser.parse_args()
    
    # Initialize scorer
    scorer = SeizureRQIScorer()
    
    # Score dataset
    results_df = scorer.score_dataset(
        vlm_csv_path=args.vlm_csv,
        gt_csv_path=args.gt_csv,
        output_path=args.output
    )
    
    print("\nScoring complete!")


if __name__ == "__main__":
    main()
