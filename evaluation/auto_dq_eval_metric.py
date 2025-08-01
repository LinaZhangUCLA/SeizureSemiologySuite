import os, math
from googleapiclient.discovery import build
from google.cloud import storage
import google.generativeai as genai
import pandas as pd
import os
import json
import csv
import time
from datetime import datetime
import traceback
import time, json, re
# from genai.errors import ServerError
import random




# GLOBAL VARIABLES
parallel_num = 3 # the num of the parallel runing scripts
parallel_index  = 2 # the first is 0


max_retries = 5
experiment = "feature_gemini_" + str(parallel_index)
feature_file = experiment + '_feature.csv'    # Output CSV (with extracted features)

all_keys = [[
"***REMOVED***",
"***REMOVED***",
"AIzaSyDQTd8OqgDrT4Ph7O36CtBq4E6PT_4Qhg",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
],
[
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
],
[
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
"***REMOVED***",
]
]
gemini_keys = all_keys[parallel_index]


CONTEXT=f"You are a seizure semiology expert analyzing doctors seizure reports. Please follow the instructions precisely."


class SeizureAnalyzer:
    def __init__(self, keys):
        self.api_keys = keys
        self.current_key_index = 0
        self.client = genai.configure(api_key = self.api_keys[self.current_key_index] )

    def _rotate_api_key(self):
        # Move to the next key (round‑robin)
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.client = genai.configure(api_key = self.api_keys[self.current_key_index] )
        self.log_processing(f"Switched to key #{self.current_key_index + 1}")

    def log_processing(self, msg):
        print(msg)
        
        
        
# Global key rotation
analyzer = SeizureAnalyzer(gemini_keys)
generation_config = {
    "temperature": 0
    # "max_output_tokens": 1024
}

def run_gemini(prompt, model_name="gemini-2.5-pro", temperature=0.2, max_output_tokens=1024):
    
    
    model = genai.GenerativeModel(model_name)
    
    max_retries=3
    for attempt in range(max_retries):
        
        try:
            # Generate content
            return model.generate_content(prompt, generation_config=genai.GenerationConfig(**generation_config))
        
        except Exception as e:
            error_msg = str(e)
            analyzer.log_processing(f"Attempt {attempt + 1} failed: {error_msg}")

            # rate-limit / quota hit?
            if any(tok in error_msg.lower() for tok in ("quota", "rate", "429", "exhausted", "key")):
                analyzer._rotate_api_key()
                analyzer.log_processing("Rotate_api_key, waiting 20s...")
        
        
# def get_events_from_behavioral_description(context=CONTEXT, reference=None, candidate=None):
    
#     prompt = f"{context}\n\n{reference}"
    
#     print(prompt)
#     print("\n")
    
#     out = run_gemini(
#         prompt=prompt,
#         model_name="gemini-2.5-pro",
#         temperature=0
#     )
    
#     print(out)
#     # return events
        
                
def auto_dq_eval_metric(context=CONTEXT, doctor=None, VLM=None):
    
    prompt = f"{context}\n\n{doctor}"
    
    print(prompt)
    print("\n")
    
    out = run_gemini(
        prompt=prompt,
        model_name="gemini-2.5-pro",
        temperature=0
    )
    
    print(out)
    
    return out
    
        
def prompt_structure(behavioral_description):
    
    full_prompt = f"""
                    Below is a description of a seizure event. The following types of events that can occur in a seizure
                    are the following:
                    
                    
                    
    
                    {behavioral_description}
                    
                    Extract all key events from the above doctor description of a patient's seizure.
                    Requirements:
                    - An event must include a symptom
                    - Every event is represented by a brief sentence with in 10 words, with a
                    subject, a predicate and optionally an object, avoid unnecessary appearance
                    descriptions.
                    - Every event must be atomic, meaning that it cannot be further split into
                    multiple events.
                    - Substitute pronouns by the nouns they refer to.
                    - Substitue acronyms for their full description

                    Please generate the response in the form of a Python dictionary string with keys
                    "events". The value of "events" is a List(str), of which each item is an event.
                    DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python
                    dictionary string. For example, your response should look like this: {{"events":
                    [event1, event2, ...]}}. DO NOT INCLUDE ANY ADDITIONAL PUNCTUATION."""
                    
    return full_prompt
     
        
def main():
    
    seizure_event_description =  """
                                    Patient is on the bed and suddenly feels an aura. 
                                    She extends her right arm and turns her head to the 
                                    right - non versive or forced. She then has gyratory 
                                    hand movements and then turns over to the right side 
                                    and around with eye deviation to the right. There are 
                                    complex body movements where she ends up moving in the bed 
                                    and climbs up. Many nurses and family members are holding her
                                    down, hence exact semiology is obscured but clearly there are 
                                    hypermotoric movements."""
    
    structure = prompt_structure(behavioral_description=seizure_event_description)
                    
    print(structure)
                    
    
    auto_dq_eval_metric(
        doctor=structure
    )
    
    
if __name__ == "__main__":
    main()