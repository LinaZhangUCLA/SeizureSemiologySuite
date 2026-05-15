"""
Seizure Report Analyzer - Single CSV Analysis (Async Version)
This script takes a single CSV file with seizure reports and analyzes each report
to extract structured information about structural completeness, temporal features,
localizing features, and penalties.
Uses async API calls for faster processing.
"""

import pandas as pd
import json
import os
from typing import Dict, List
import asyncio
from openai import AsyncOpenAI
from tqdm import tqdm


class SeizureReportAnalyzer:
    
    def __init__(self, api_key: str = None, model: str = "qwen-plus-latest"):
        """Initialize the analyzer with dashscope API (Qwen) - Async version"""
        self.api_key = api_key or os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("Missing API key. Pass --api_key or set QWEN_API_KEY/DASHSCOPE_API_KEY.")
        self.model = model
        self.features = [
            'blank_stare', 'close_eyes', 'eye_blinking',
            'tonic', 'clonic', 'arm_flexion', 'arm_straightening', 'figure4', 'oral_automatisms', 'limb_automatisms',
            'face_pulling', 'face_twitching', 'head_turning', 'asynchronous_movement', 'pelvic_thrusting',
            'arms_move_simultaneously', 'full_body_shaking', 'ictal_vocalization', 'verbal_responsiveness', 'occur_during_sleep',
        ]
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    
    def create_analysis_prompt(self, report: str) -> str:
        """Create a prompt to analyze a single seizure report"""
        features_list = ', '.join(self.features)
        
        prompt = f"""You are an expert neurologist analyzing a seizure semiology report.
Analyze the following seizure report and extract structured information:

SEIZURE REPORT:
{report}

Please analyze and extract:
1. STRUCTURAL COMPLETENESS:
   List which of the following components are actually mentioned in the report (only include present ones):
   - ONSET: Initial visible signs at seizure beginning (first motor signs, head turning, eye movements, blank stare, etc.)
   - PROPAGATION: Evolution/spread of seizure activity - any description of how symptoms evolve, spread to other body parts, or involve multiple motor activities (e.g., automatisms, bilateral movements, tonic-clonic activity, asymmetric posturing)
   - POSTICTAL: State immediately after seizure ends (confusion, unresponsiveness, amnesia, drowsiness, inability to recall)
   Return as a list of strings, e.g. ["onset", "propagation"] or [] if none present.
   
   **Note**: If the report describes multiple motor activities or symptoms involving different body parts, it likely includes propagation.

2. TEMPORAL EVENT SEQUENCE:
   Extract a chronologically-ordered list of seizure features/events from the report.
   **IMPORTANT: You MUST ONLY use features from this vocabulary list (20 features):**
   {features_list}
   
   List features in the order they occur.
   Each feature should appear at most once in its temporal position.
   Do NOT add any features that are not in the above vocabulary list.

3. KEY LOCALIZING FEATURES:
   Extract the specific values for:
   - Head turning: Direction of head turning ("left", "right", or "none" if not mentioned)
   - FIRST MOTOR LATERALITY: Laterality of first motor sign ("left", "right", or "none" if not mentioned)
   - ASYMMETRIC POSTURING: Type of asymmetric tonic arm posturing ("right arm flexion", "left arm extension", "left arm flexion", "right arm extension", or "none" if not mentioned)
   - FIGURE-4: Figure-4 sign arm extension ("left", "right", or "none" if not mentioned)


4. CONTENT ANALYSIS:
   - SAFETY ISSUES: Does the report contain any hazardous medical recommendations? (true/false)
   - NON-SEIZURE CONTENT: Does the report incorrectly include nursing interventions, repositioning, or device alarms? (true/false)

Return your analysis in this JSON format:
{{
    "structural_completeness": ["onset", "postictal"],
    "event_sequence": ["<ordered list of seizure features in temporal order>"],
    "key_localizing_features": {{
        "head_turning": "left/right/none",
        "first_motor_laterality": "left/right/none",
        "asymmetric_posturing": "right arm flexion/left arm extension/left arm flexion/right arm extension/none",
        "figure_4": "left/right/none"
    }},
    "content_analysis": {{
        "has_safety_issues": true/false,
        "includes_non_seizure": true/false
    }}
}}"""
        return prompt
    
    def parse_llm_response(self, response: str) -> Dict:
        """Parse the LLM response into structured data"""
        try:
            # Extract JSON from response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```", 1)[0]
            else:
                json_str = response
            analysis = json.loads(json_str)
            return analysis
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return None
    
    async def analyze_single_report(self, report: str, file_name: str) -> Dict:
        """Analyze a single seizure report asynchronously"""
        prompt = self.create_analysis_prompt(report)
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert neurologist analyzer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            llm_response = response.choices[0].message.content
            analysis = self.parse_llm_response(llm_response)
            
            if analysis is not None:
                # Structural completeness as a list
                struct_list = analysis.get('structural_completeness', [])
                struct_list_str = json.dumps(struct_list, ensure_ascii=False)

                # Event sequence
                event_seq = analysis.get('event_sequence', [])
                event_seq_str = json.dumps(event_seq, ensure_ascii=False)

                # Localizing features - combine into one JSON string
                loc_features = analysis.get('key_localizing_features', {})
                loc_features_str = json.dumps(loc_features, ensure_ascii=False)

                # Content analysis - combine into one JSON string
                content = analysis.get('content_analysis', {})
                content_str = json.dumps(content, ensure_ascii=False)

                # Report length - calculate directly from report
                report_length = len(str(report))

                result = {
                    'file_name': file_name,
                    'structural_completeness': struct_list_str,
                    'event_sequence': event_seq_str,
                    'localizing_features': loc_features_str,
                    'content_analysis': content_str,
                    'report_length': report_length
                }
                return result
            else:
                # If analysis fails, add empty result
                return {
                    'file_name': file_name,
                    'structural_completeness': '[]',
                    'event_sequence': '[]',
                    'localizing_features': '{}',
                    'content_analysis': '{}',
                    'report_length': len(str(report))
                }
        except Exception as e:
            tqdm.write(f"❌ Error {file_name}: {str(e)[:50]}")
            return {
                'file_name': file_name,
                'structural_completeness': '[]',
                'event_sequence': '[]',
                'localizing_features': '{}',
                'content_analysis': '{}',
                'report_length': len(str(report))
            }
    
    async def analyze_dataset(self, csv_path: str, output_path: str = None, batch_size: int = 20) -> pd.DataFrame:
        """Analyze all reports in CSV with async batch processing and incremental saving"""
        # Load data
        df = pd.read_csv(csv_path)
        
        if 'report' not in df.columns:
            raise ValueError("CSV must contain a 'report' column")
        
        # Check if output file exists and load processed files
        processed_files = set()
        if output_path and os.path.exists(output_path):
            try:
                existing_df = pd.read_csv(output_path)
                processed_files = set(existing_df['file_name'].tolist())
                print(f"Found {len(processed_files)} already processed files, skipping them...")
            except Exception as e:
                print(f"Could not load existing results: {e}")
        
        # Filter out already processed files
        df_to_process = df[~df.get('file_name', pd.Series([''] * len(df))).isin(processed_files)]
        
        if len(df_to_process) == 0:
            print("All files already processed!")
            if output_path and os.path.exists(output_path):
                return pd.read_csv(output_path)
            else:
                return pd.DataFrame()
        
        print(f"Analyzing {len(df_to_process)} reports with batch size {batch_size}...")
        print(f"Skipped {len(processed_files)} already processed files.")
        
        # Process in batches and save incrementally
        all_results = []
        total_batches = (len(df_to_process)-1)//batch_size + 1
        
        with tqdm(total=len(df_to_process), desc="Processing reports", unit="report") as pbar:
            for i in range(0, len(df_to_process), batch_size):
                batch_df = df_to_process.iloc[i:i+batch_size]
                batch_num = i//batch_size + 1
                
                # Create async tasks for this batch
                tasks = [
                    self.analyze_single_report(row['report'], row.get('file_name', f'report_{idx}'))
                    for idx, row in batch_df.iterrows()
                ]
                
                batch_results = await asyncio.gather(*tasks)
                all_results.extend(batch_results)
                
                # Update progress bar
                pbar.update(len(batch_results))
                pbar.set_postfix({'batch': f'{batch_num}/{total_batches}'})
                
                # Save to CSV after each batch
                if output_path:
                    batch_results_df = pd.DataFrame(batch_results)
                    if os.path.exists(output_path):
                        # Append to existing file
                        batch_results_df.to_csv(output_path, mode='a', header=False, index=False)
                    else:
                        # Create new file with header
                        batch_results_df.to_csv(output_path, mode='w', header=True, index=False)
                
                # Small delay between batches
                if i + batch_size < len(df_to_process):
                    await asyncio.sleep(1)
        
        # Create final results dataframe
        results_df = pd.DataFrame(all_results)
        
        # Calculate summary statistics
        print("\n=== Analysis Summary ===")
        print(f"Total reports analyzed in this run: {len(results_df)}")
        if len(results_df) > 0:
            print(f"Average report length: {results_df['report_length'].mean():.1f} characters")
        
        # Load complete results if file exists
        if output_path and os.path.exists(output_path):
            complete_df = pd.read_csv(output_path)
            print(f"Total reports in output file: {len(complete_df)}")
            return complete_df
        
        return results_df


def main():
    """Main function to run the analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze seizure reports from a single CSV file (Async)')
    parser.add_argument('--input_csv', type=str, required=True,
                       help='Path to input CSV file with seizure reports')
    parser.add_argument('--output', type=str, default='seizure_analysis_results.csv',
                       help='Output path for results CSV')
    parser.add_argument('--api_key', type=str, default=None,
                       help='API key (optional if set in environment)')
    parser.add_argument('--model', type=str, default='qwen-plus-latest',
                       help='Qwen model to use for analysis')
    parser.add_argument('--batch_size', type=int, default=20,
                       help='Number of reports per batch for incremental saving')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SeizureReportAnalyzer(api_key=args.api_key, model=args.model)
    
    # Run analysis with asyncio
    results_df = asyncio.run(analyzer.analyze_dataset(
        csv_path=args.input_csv,
        output_path=args.output,
        batch_size=args.batch_size
    ))
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
