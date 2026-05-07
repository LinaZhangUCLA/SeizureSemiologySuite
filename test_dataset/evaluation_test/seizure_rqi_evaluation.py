"""
SeizureRQI (Seizure Report Quality Index) Evaluation Script
This script evaluates VLM-generated seizure reports against ground truth annotations
using an LLM-as-judge approach based on multiple criteria.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json
import os
from dataclasses import dataclass
from openai import OpenAI
import time
from tqdm import tqdm

@dataclass
class EvaluationCriteria:
    """Evaluation criteria with weights"""
    structural_completeness_weight: float = 0.25  # S
    symptom_coverage_weight: float = 0.25  # C
    key_localizing_features_weight: float = 0.25  # L
    temporal_fidelity_weight: float = 0.25  # T

@dataclass
class SeizureFeatures:
    """20 key seizure features for evaluation"""
    features = [
        "blank_stare", "lip_smacking", "versive_head_turn", "unilateral_arm_movement",
        "bilateral_tonic_clonic", "automatisms", "eye_deviation", "facial_pulling",
        "dystonic_posturing", "figure_four_sign", "ictal_cry", "oral_automatisms",
        "manual_automatisms", "leg_automatisms", "consciousness_alteration",
        "postictal_confusion", "postictal_amnesia", "respiratory_changes",
        "vocalization", "asymmetric_movements"
    ]

class SeizureRQIEvaluator:
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        """Initialize the evaluator with OpenAI API"""
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.criteria = EvaluationCriteria()
        self.features = SeizureFeatures.features
        
    def create_evaluation_prompt(self, vlm_report: str, ground_truth: str) -> str:
        """Create a detailed prompt for LLM evaluation"""
        prompt = f"""You are an expert neurologist evaluating seizure semiology reports. 
Compare the following VLM-generated report against the ground truth annotation and provide scores.

GROUND TRUTH ANNOTATION:
{ground_truth}

VLM-GENERATED REPORT:
{vlm_report}

Please evaluate based on these criteria:

1. STRUCTURAL COMPLETENESS (S) - Score 0-100:
   - ONSET: Does the report describe concrete visible signs at seizure beginning? 
     Look for: blank stare, lip smacking, versive head turn, unilateral arm flexion/extension
   - PROPAGATION: Does it describe evolution/spread and laterality?
     Look for: fencer posturing (L flexion/R extension), transition to bilateral tonic-clonic
   - POSTICTAL: Does it describe immediate state after movements stop?
     Look for: confusion, unresponsiveness, amnesia, drowsiness

2. SYMPTOM COVERAGE (C) - Score 0-100:
   Evaluate coverage of these 20 key features:
   {', '.join(self.features)}
   
   Count how many features are accurately described in the VLM report compared to ground truth.

3. KEY LOCALIZING FEATURES (L) - Score 0-100:
   Focus on high-impact lateralizing elements:
   - Head version direction (left/right)
   - First motor sign laterality
   - Asymmetric tonic posturing (which arm extends/flexes)
   - Figure-4 sign accuracy
   
4. TEMPORAL & QUANTITATIVE FIDELITY (T) - Score 0-100:
   - Correct sequence of events
   - Accurate duration mentions (if present)
   - Proper temporal relationships between symptoms

5. PENALTIES:
   - HALLUCINATION: List any features described that are NOT in ground truth (-10 per hallucination)
   - SAFETY: Any hazardous medical recommendations? (yes/no)
   - NON-SEIZURE: Did report incorrectly include nursing interventions, repositioning, or device alarms? (yes/no)

Return your evaluation in this JSON format:
{{
    "structural_completeness": {{
        "onset_score": <0-100>,
        "propagation_score": <0-100>,
        "postictal_score": <0-100>,
        "overall_score": <0-100>,
        "justification": "<brief explanation>"
    }},
    "symptom_coverage": {{
        "features_found": ["<list of correctly identified features>"],
        "features_missed": ["<list of missed features>"],
        "coverage_ratio": <ratio>,
        "score": <0-100>,
        "justification": "<brief explanation>"
    }},
    "key_localizing_features": {{
        "head_version_correct": <true/false>,
        "first_motor_laterality_correct": <true/false>,
        "asymmetric_posturing_correct": <true/false>,
        "figure_4_correct": <true/false>,
        "score": <0-100>,
        "justification": "<brief explanation>"
    }},
    "temporal_fidelity": {{
        "sequence_accuracy": <0-100>,
        "duration_accuracy": <0-100>,
        "temporal_relations": <0-100>,
        "score": <0-100>,
        "justification": "<brief explanation>"
    }},
    "penalties": {{
        "hallucinated_features": ["<list of hallucinated features>"],
        "hallucination_count": <number>,
        "has_safety_issues": <true/false>,
        "includes_non_seizure": <true/false>,
        "penalty_multiplier": <calculated penalty 0-1>
    }}
}}"""
        return prompt

    def parse_llm_response(self, response: str) -> Dict:
        """Parse the LLM response into structured scores"""
        try:
            # Extract JSON from response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            else:
                json_str = response
            
            return json.loads(json_str)
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return None

    def calculate_seizure_rqi(self, evaluation: Dict) -> float:
        """Calculate final SeizureRQI score based on evaluation results"""
        # Extract component scores
        s_score = evaluation["structural_completeness"]["overall_score"]
        c_score = evaluation["symptom_coverage"]["score"]
        l_score = evaluation["key_localizing_features"]["score"]
        t_score = evaluation["temporal_fidelity"]["score"]
        
        # Calculate weighted sum
        weighted_score = (
            self.criteria.structural_completeness_weight * s_score +
            self.criteria.symptom_coverage_weight * c_score +
            self.criteria.key_localizing_features_weight * l_score +
            self.criteria.temporal_fidelity_weight * t_score
        )
        
        # Apply penalties
        penalties = evaluation["penalties"]
        
        # Hallucination penalty (multiplicative, -10% per hallucination)
        hallucination_penalty = max(0, 1 - (penalties["hallucination_count"] * 0.1))
        
        # Safety gate (cap at 50 if safety issues)
        if penalties["has_safety_issues"]:
            weighted_score = min(weighted_score, 50)
        
        # Non-seizure content penalty (reduce by 20% if included)
        if penalties["includes_non_seizure"]:
            weighted_score *= 0.8
        
        # Apply hallucination penalty
        final_score = weighted_score * hallucination_penalty
        
        return min(100, max(0, final_score))  # Ensure score is between 0-100

    def evaluate_single_report(self, vlm_report: str, ground_truth: str) -> Tuple[float, Dict]:
        """Evaluate a single VLM report against ground truth"""
        prompt = self.create_evaluation_prompt(vlm_report, ground_truth)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert neurologist evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            llm_response = response.choices[0].message.content
            evaluation = self.parse_llm_response(llm_response)
            
            if evaluation:
                seizure_rqi = self.calculate_seizure_rqi(evaluation)
                return seizure_rqi, evaluation
            else:
                return None, None
                
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return None, None

    def evaluate_dataset(self, vlm_csv_path: str, ground_truth_csv_path: str, 
                        output_path: str = None) -> pd.DataFrame:
        """Evaluate entire dataset and save results"""
        # Load data
        vlm_df = pd.read_csv(vlm_csv_path)
        gt_df = pd.read_csv(ground_truth_csv_path)
        
        # Merge on file_name
        merged_df = pd.merge(
            vlm_df[['file_name', 'report']], 
            gt_df[['file_name', 'report']], 
            on='file_name', 
            suffixes=('_vlm', '_gt')
        )
        
        results = []
        
        print(f"Evaluating {len(merged_df)} reports...")
        for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
            vlm_report = row['report_vlm']
            ground_truth = row['report_gt']
            file_name = row['file_name']
            
            seizure_rqi, evaluation_details = self.evaluate_single_report(
                vlm_report, ground_truth
            )
            
            if seizure_rqi is not None:
                result = {
                    'file_name': file_name,
                    'seizure_rqi': seizure_rqi,
                    'structural_completeness': evaluation_details['structural_completeness']['overall_score'],
                    'symptom_coverage': evaluation_details['symptom_coverage']['score'],
                    'key_localizing_features': evaluation_details['key_localizing_features']['score'],
                    'temporal_fidelity': evaluation_details['temporal_fidelity']['score'],
                    'hallucination_count': evaluation_details['penalties']['hallucination_count'],
                    'has_safety_issues': evaluation_details['penalties']['has_safety_issues'],
                    'includes_non_seizure': evaluation_details['penalties']['includes_non_seizure'],
                    'evaluation_details': json.dumps(evaluation_details)
                }
                results.append(result)
            
            # Rate limiting
            time.sleep(0.5)
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Calculate summary statistics
        summary_stats = {
            'mean_seizure_rqi': results_df['seizure_rqi'].mean(),
            'std_seizure_rqi': results_df['seizure_rqi'].std(),
            'mean_structural': results_df['structural_completeness'].mean(),
            'mean_symptom': results_df['symptom_coverage'].mean(),
            'mean_localizing': results_df['key_localizing_features'].mean(),
            'mean_temporal': results_df['temporal_fidelity'].mean(),
            'total_hallucinations': results_df['hallucination_count'].sum(),
            'reports_with_safety_issues': results_df['has_safety_issues'].sum(),
            'reports_with_non_seizure': results_df['includes_non_seizure'].sum()
        }
        
        print("\n=== Evaluation Summary ===")
        for key, value in summary_stats.items():
            print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
        
        # Save results
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")
            
            # Save summary stats
            summary_path = output_path.replace('.csv', '_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary_stats, f, indent=2)
            print(f"Summary saved to: {summary_path}")
        
        return results_df


def main():
    """Main function to run the evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate VLM seizure reports using SeizureRQI')
    parser.add_argument('--vlm_csv', type=str, required=True, 
                       help='Path to VLM inference results CSV')
    parser.add_argument('--gt_csv', type=str, required=True,
                       help='Path to ground truth annotations CSV')
    parser.add_argument('--output', type=str, default='seizure_rqi_results.csv',
                       help='Output path for results')
    parser.add_argument('--api_key', type=str, default=None,
                       help='OpenAI API key (optional if set in environment)')
    parser.add_argument('--model', type=str, default='gpt-4o',
                       help='OpenAI model to use for evaluation')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = SeizureRQIEvaluator(api_key=args.api_key, model=args.model)
    
    # Run evaluation
    results_df = evaluator.evaluate_dataset(
        vlm_csv_path=args.vlm_csv,
        ground_truth_csv_path=args.gt_csv,
        output_path=args.output
    )
    
    print("\nEvaluation complete!")
    

if __name__ == "__main__":
    main()