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
import time
from tqdm import tqdm
from openai import OpenAI
import re
import Levenshtein
try:
    from sklearn.metrics import f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. F1 score calculation will be simplified.")

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

    def __init__(self, api_key: str = None, model: str = "qwen-plus-latest"):
        """Initialize the evaluator with OpenAI SDK (dashscope compatible mode)"""
        self.api_key = api_key or os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        self.model = model
        self.criteria = EvaluationCriteria()
        self.features = SeizureFeatures.features
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def tokenize(self, seq_list):
        # seq_list: list[str] from LLM output
        tokens = []
        for p in seq_list:
            t = str(p).strip().strip('"').strip("'").strip()
            t = t.rstrip('.;')
            t = re.sub(r"\s+", "_", t)
            if t:
                tokens.append(t)
        return tokens

    def align_for_f1(self, pred_tokens, gt_tokens):
        y_true = []
        y_pred = []
        for op in Levenshtein.opcodes(pred_tokens, gt_tokens):
            tag = op[0]
            s1, e1, s2, e2 = op[1], op[2], op[3], op[4]
            if tag == "equal":
                y_pred.extend(pred_tokens[s1:e1])
                y_true.extend(gt_tokens[s2:e2])
            elif tag == "replace":
                for i in range(e1 - s1):
                    y_pred.append(pred_tokens[s1 + i])
                    y_true.append(gt_tokens[s2 + i])
            elif tag == "delete":
                for i in range(e1 - s1):
                    y_pred.append("<GAP>")
                    y_true.append(gt_tokens[s2] if s2 < len(gt_tokens) else "<GAP>")
            elif tag == "insert":
                for i in range(e2 - s2):
                    y_pred.append(pred_tokens[s1] if s1 < len(pred_tokens) else "<GAP>")
                    y_true.append(gt_tokens[s2 + i])
        n = min(len(y_true), len(y_pred))
        return y_true[:n], y_pred[:n]

        
    def create_evaluation_prompt(self, vlm_report: str, ground_truth: str) -> str:
        """Create a detailed prompt for LLM evaluation"""
        features_list = ', '.join(self.features)
        
        prompt_template = """You are an expert neurologist evaluating seizure semiology reports.
Compare the following VLM-generated report against the ground truth annotation and provide structured outputs.

GROUND TRUTH ANNOTATION:
{ground_truth}

VLM-GENERATED REPORT:
{vlm_report}

Please evaluate based on these criteria:

1. STRUCTURAL COMPLETENESS (S):
     For each of the following items, indicate whether it is present in the ground truth (gt) and in the VLM report (vlm):
     - ONSET: Does the report describe concrete visible signs at seizure beginning?
         Look for: blank stare, lip smacking, versive head turn, unilateral arm flexion/extension
     - PROPAGATION: Does it describe evolution/spread and laterality?
         Look for: fencer posturing (L flexion/R extension), transition to bilateral tonic-clonic
     - POSTICTAL: Does it describe immediate state after movements stop?
         Look for: confusion, unresponsiveness, amnesia, drowsiness

2. SYMPTOM COVERAGE (C):
     Evaluate coverage of these 20 key features:
     {features_list}
     In your JSON output, include three lists under "symptom_coverage":
         - "features_found_gt": list of key features present in the ground truth annotation
         - "features_found_vlm": list of key features present in the VLM report
         - "features_missed": list of features present in ground truth annotation but missing in the VLM report (should be computed as features_found_gt minus features_found_vlm)

3. KEY LOCALIZING FEATURES (L):
     For each of the following items, indicate whether it is present in the ground truth (gt) and in the VLM report (vlm):
     - HEAD VERSION: Head version direction (left/right)
     - FIRST MOTOR LATERALITY: First motor sign laterality
     - ASYMMETRIC POSTURING: Asymmetric tonic posturing (which arm extends/flexes)
     - FIGURE-4: Figure-4 sign accuracy

4. TEMPORAL FIDELITY (T):
     For this criterion, do NOT return a numeric score. Instead, in your JSON output, include:
         - "temporal_fidelity": {{
                 "vlm_event_sequence": ["<ordered list of events as described in the VLM report>"],
                 "gt_event_sequence": ["<ordered list of events as described in the ground truth>"]
             }}

5. PENALTIES:
     For each of the following items, indicate whether it is present in the ground truth (gt) and in the VLM report (vlm):
     - HALLUCINATED FEATURES: List any features described that are NOT in ground truth (-10 per hallucination)
     - SAFETY: Any hazardous medical recommendations? (yes/no)
     - NON-SEIZURE: Did report incorrectly include nursing interventions, repositioning, or device alarms? (yes/no)

Return your evaluation in this JSON format:
{{
    "structural_completeness": {{
        "onset": {{"gt": true/false, "vlm": true/false}},
        "propagation": {{"gt": true/false, "vlm": true/false}},
        "postictal": {{"gt": true/false, "vlm": true/false}},
        "justification": "<brief explanation>"
    }},
    "symptom_coverage": {{
        "features_found_gt": ["<list of features present in ground truth annotation>"],
        "features_found_vlm": ["<list of features present in VLM report>"],
        "features_missed": ["<list of features present in ground truth annotation but missing in VLM report>"]
    }},
    "key_localizing_features": {{
        "head_version": {{"gt": true/false, "vlm": true/false}},
        "first_motor_laterality": {{"gt": true/false, "vlm": true/false}},
        "asymmetric_posturing": {{"gt": true/false, "vlm": true/false}},
        "figure_4": {{"gt": true/false, "vlm": true/false}},
        "justification": "<brief explanation>"
    }},
    "temporal_fidelity": {{
        "vlm_event_sequence": ["<ordered list of events as described in the VLM report>"],
        "gt_event_sequence": ["<ordered list of events as described in the ground truth>"]
    }},
    "penalties": {{
        "hallucinated_features": {{"gt": true/false, "vlm": true/false}},
        "has_safety_issues": {{"gt": true/false, "vlm": true/false}},
        "includes_non_seizure": {{"gt": true/false, "vlm": true/false}},
        "hallucination_count": <number>,
        "penalty_multiplier": <calculated penalty 0-1>
    }}
}}"""

        prompt = prompt_template.format(
            ground_truth=ground_truth,
            vlm_report=vlm_report,
            features_list=features_list
        )
        return prompt

    def parse_llm_response(self, response: str) -> Dict:
        """Parse the LLM response into structured scores"""
        try:
            # Extract JSON from response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```", 1)[0]
            else:
                json_str = response
            evaluation = json.loads(json_str)
            # Fallback: if lengths are missing, compute them
            if "vlm_report_len" not in evaluation:
                evaluation["vlm_report_len"] = len(str(evaluation.get("vlm_report", "")))
            if "gt_report_len" not in evaluation:
                evaluation["gt_report_len"] = len(str(evaluation.get("gt_report", "")))
            return evaluation
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return None


    def calculate_penalty(self, penalties: Dict, vlm_report_len: int, gt_report_len: int) -> float:
        """Calculate penalty multiplier based on hallucinations, safety, non-seizure, and length."""
        # Hallucination penalty (multiplicative, -10% per hallucination)
        hallucination_penalty = max(0, 1 - (penalties.get("hallucination_count", 0) * 0.1))
        # Safety gate (cap at 50 if safety issues)
        safety_gate = penalties.get("has_safety_issues", False)
        # Non-seizure content penalty (reduce by 20% if included)
        non_seizure_penalty = 0.8 if penalties.get("includes_non_seizure", False) else 1.0
        # Length penalty: every 50 chars over GT, -5 points
        length_penalty = 1.0
        if vlm_report_len > gt_report_len:
            over_len = vlm_report_len - gt_report_len
            length_penalty -= (over_len // 50) * 0.05
            length_penalty = max(0, length_penalty)
        # Combine all penalties
        penalty_multiplier = hallucination_penalty * non_seizure_penalty * length_penalty
        return penalty_multiplier, safety_gate

    def calculate_seizure_rqi(self, evaluation: Dict) -> float:
        """Calculate final SeizureRQI score based on evaluation results, using explicit report lengths and penalty function."""
        # 1. STRUCTURAL COMPLETENESS: count gt true, count vlm true that matches gt true
        struct_keys = ["onset", "propagation", "postictal"]
        gt_struct_true = sum([evaluation["structural_completeness"].get(k, {}).get("gt", False) for k in struct_keys])
        vlm_struct_true = sum([evaluation["structural_completeness"].get(k, {}).get("gt", False) and evaluation["structural_completeness"].get(k, {}).get("vlm", False) for k in struct_keys])
        s_score = (vlm_struct_true / gt_struct_true) * 100 if gt_struct_true > 0 else 0

        # 2. SYMPTOM COVERAGE: VLM features_found_vlm / GT features_found_gt
        try:
            features_found_vlm = set(evaluation["symptom_coverage"].get("features_found_vlm", []))
            features_found_gt = set(evaluation["symptom_coverage"].get("features_found_gt", []))
            num_vlm = len(features_found_vlm)
            num_gt = len(features_found_gt)
            c_score = (num_vlm / num_gt) * 100 if num_gt > 0 else 0
        except Exception:
            c_score = 0

        # 3. KEY LOCALIZING FEATURES: count gt true, count vlm true that matches gt true
        loc_keys = ["head_version", "first_motor_laterality", "asymmetric_posturing", "figure_4"]
        gt_loc_true = sum([evaluation["key_localizing_features"].get(k, {}).get("gt", False) for k in loc_keys])
        vlm_loc_true = sum([evaluation["key_localizing_features"].get(k, {}).get("gt", False) and evaluation["key_localizing_features"].get(k, {}).get("vlm", False) for k in loc_keys])
        l_score = (vlm_loc_true / gt_loc_true) * 100 if gt_loc_true > 0 else 0

        # 4. TEMPORAL FIDELITY: use F1 score between event sequences
        try:
            vlm_seq = evaluation["temporal_fidelity"].get("vlm_event_sequence", [])
            gt_seq = evaluation["temporal_fidelity"].get("gt_event_sequence", [])
            vlm_tokens = self.tokenize(vlm_seq)
            gt_tokens = self.tokenize(gt_seq)
            if len(vlm_tokens) == 0 and len(gt_tokens) == 0:
                f1 = 1.0
            else:
                y_true, y_pred = self.align_for_f1(vlm_tokens, gt_tokens)
                try:
                    if SKLEARN_AVAILABLE:
                        f1 = float(f1_score(y_true, y_pred, average="micro"))
                    else:
                        # Simple F1 calculation if sklearn not available
                        if len(y_true) == len(y_pred):
                            matches = sum(1 for a, b in zip(y_true, y_pred) if a == b)
                            f1 = matches / len(y_true) if len(y_true) > 0 else 0.0
                        else:
                            f1 = 0.0
                except Exception:
                    f1 = 0.0
            t_score = f1 * 100
        except Exception:
            t_score = 0

        # Explicitly record report lengths
        vlm_report_len = len(str(evaluation.get("vlm_report", "")))
        gt_report_len = len(str(evaluation.get("gt_report", "")))

        # Calculate weighted sum
        weighted_score = (
            self.criteria.structural_completeness_weight * s_score +
            self.criteria.symptom_coverage_weight * c_score +
            self.criteria.key_localizing_features_weight * l_score +
            self.criteria.temporal_fidelity_weight * t_score
        )

        # Calculate penalty multiplier
        penalties = evaluation["penalties"]
        penalty_multiplier, safety_gate = self.calculate_penalty(penalties, vlm_report_len, gt_report_len)

        # Safety gate (cap at 50 if safety issues)
        if safety_gate:
            weighted_score = min(weighted_score, 50)

        # Apply penalty multiplier
        final_score = weighted_score * penalty_multiplier
        return min(100, max(0, final_score))  # Ensure score is between 0-100

    def evaluate_single_report(self, vlm_report: str, ground_truth: str) -> Tuple[float, Dict]:
        """Evaluate a single VLM report against ground truth using OpenAI SDK (dashscope compatible mode)"""
        prompt = self.create_evaluation_prompt(vlm_report, ground_truth)
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert neurologist evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            llm_response = completion.choices[0].message.content
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
        
        print(f"Evaluating {len(merged_df)} reports...", flush=True)
        for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
            vlm_report = row['report_vlm']
            ground_truth = row['report_gt']
            file_name = row['file_name']
            
            seizure_rqi, evaluation_details = self.evaluate_single_report(
                vlm_report, ground_truth
            )
            
            if seizure_rqi is not None:
          
                struct_keys = ["onset", "propagation", "postictal"]
                gt_struct_true = sum([evaluation_details["structural_completeness"].get(k, {}).get("gt", False) for k in struct_keys]) if evaluation_details else 0
                vlm_struct_true = sum([evaluation_details["structural_completeness"].get(k, {}).get("gt", False) and evaluation_details["structural_completeness"].get(k, {}).get("vlm", False) for k in struct_keys]) if evaluation_details else 0
                s_score = (vlm_struct_true / gt_struct_true) * 100 if gt_struct_true > 0 else 0
                
                features_found_vlm = set(evaluation_details['symptom_coverage'].get('features_found_vlm', [])) if evaluation_details else set()
                features_found_gt = set(evaluation_details['symptom_coverage'].get('features_found_gt', [])) if evaluation_details else set()
                num_vlm = len(features_found_vlm)
                num_gt = len(features_found_gt)
                symptom_coverage = (num_vlm / num_gt) * 100 if num_gt > 0 else 0
                
                loc_keys = ["head_version", "first_motor_laterality", "asymmetric_posturing", "figure_4"]
                gt_loc_true = sum([evaluation_details['key_localizing_features'].get(k, {}).get("gt", False) for k in loc_keys]) if evaluation_details else 0
                vlm_loc_true = sum([evaluation_details['key_localizing_features'].get(k, {}).get("gt", False) and evaluation_details['key_localizing_features'].get(k, {}).get("vlm", False) for k in loc_keys]) if evaluation_details else 0
                key_localizing_features = (vlm_loc_true / gt_loc_true) * 100 if gt_loc_true > 0 else 0

                vlm_seq = evaluation_details['temporal_fidelity'].get('vlm_event_sequence', []) if evaluation_details else []
                gt_seq = evaluation_details['temporal_fidelity'].get('gt_event_sequence', []) if evaluation_details else []
                vlm_tokens = self.tokenize(vlm_seq)
                gt_tokens = self.tokenize(gt_seq)
                if len(vlm_tokens) == 0 and len(gt_tokens) == 0:
                    temporal_fidelity = 100.0
                else:
                    y_true, y_pred = self.align_for_f1(vlm_tokens, gt_tokens)
                    try:
                        if SKLEARN_AVAILABLE:
                            from sklearn.metrics import f1_score
                            f1 = float(f1_score(y_true, y_pred, average="micro"))
                        else:
                            if len(y_true) == len(y_pred):
                                matches = sum(1 for a, b in zip(y_true, y_pred) if a == b)
                                f1 = matches / len(y_true) if len(y_true) > 0 else 0.0
                            else:
                                f1 = 0.0
                    except Exception:
                        f1 = 0.0
                    temporal_fidelity = f1 * 100
    
                has_safety_issues = evaluation_details['penalties']['has_safety_issues'] if evaluation_details else None
                if isinstance(has_safety_issues, dict):
    
                    has_safety_issues = has_safety_issues.get('vlm', False) if 'vlm' in has_safety_issues else bool(has_safety_issues)
                elif not isinstance(has_safety_issues, (bool, int)):
                    has_safety_issues = bool(has_safety_issues)
   
                includes_non_seizure = evaluation_details['penalties']['includes_non_seizure'] if evaluation_details else None
                if isinstance(includes_non_seizure, dict):
                    includes_non_seizure = includes_non_seizure.get('vlm', False) if 'vlm' in includes_non_seizure else bool(includes_non_seizure)
                elif not isinstance(includes_non_seizure, (bool, int)):
                    includes_non_seizure = bool(includes_non_seizure)
                result = {
                    'file_name': file_name,
                    'seizure_rqi': seizure_rqi,
                    'structural_completeness': s_score,
                    'symptom_coverage': symptom_coverage,
                    'key_localizing_features': key_localizing_features,
                    'temporal_fidelity': temporal_fidelity,
                    'hallucination_count': evaluation_details['penalties']['hallucination_count'] if evaluation_details else None,
                    'has_safety_issues': has_safety_issues,
                    'includes_non_seizure': includes_non_seizure,
                    'evaluation_details': json.dumps(evaluation_details, ensure_ascii=False) if evaluation_details else None
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
        
        print("\n=== Evaluation Summary ===", flush=True)
        for key, value in summary_stats.items():
            try:
                print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}", flush=True)
            except Exception:
                print(f"{key}: {str(value).encode('utf-8', errors='ignore').decode('utf-8')}", flush=True)
        
        # Save results
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}", flush=True)
            
            # Save summary stats
            summary_path = output_path.replace('.csv', '_summary.json')
            # 转换所有 numpy/pandas 类型为原生类型
            def to_native(val):
                if hasattr(val, 'item'):
                    return val.item()
                if isinstance(val, (np.generic, np.ndarray)):
                    return val.tolist()
                return val
            summary_stats_native = {k: to_native(v) for k, v in summary_stats.items()}
            with open(summary_path, 'w') as f:
                json.dump(summary_stats_native, f, indent=2)
            print(f"Summary saved to: {summary_path}", flush=True)
        
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
    parser.add_argument('--model', type=str, default='qwen-plus-latest',
                       help='Qwen model to use for evaluation')
    
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