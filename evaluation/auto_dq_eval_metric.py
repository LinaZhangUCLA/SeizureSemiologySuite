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
import json
import sys
from typing import List, Dict, Union



# GLOBAL VARIABLES
parallel_num = 3 # the num of the parallel runing scripts
parallel_index  = 2 # the first is 0


max_retries = 5
experiment = "feature_gemini_" + str(parallel_index)
feature_file = experiment + '_feature.csv'    # Output CSV (with extracted features)

all_keys = [[
"AIzaSyBIVWa312CvfQzDm5dMYuZwo2iGLoUbN-U",
"AIzaSyCYPgpXfXDVsklKOydVgApb8ChXdXdP610",
"AIzaSyDQTd8OqgDrT4Ph7O36CtBq4E6PT_4Qhg",
"AIzaSyBBls6SnNQrV2TNvYAJZgmzLFZU5GTil3U",
"AIzaSyAK84faBFnxj3QUDDvWXYBxjWODpTnjc1c",
"AIzaSyCFRsx-hr43phPG0GLgJUVPW-WgHnbRxd4",
"AIzaSyD0VNimrZxlGnics7blvd9ZXB8ujvy2thU",
"AIzaSyD5U-4WUrrmdQ0-YkOh8TR05nsHMYVLwuo",
"AIzaSyAnRUL3LWGkyr9_VXJhqcXQTrf4yMmIo80",
"AIzaSyBYaMSgezO5adN-XbAoKJf3Vsw4ld2B7Dc",
"AIzaSyAeSTzaLVlYYHYStApnyrft8W4rn-9wJTE",
"AIzaSyAkcB7j59YHThPefpl1QNiHU2EmAaceyKo",
"AIzaSyCHuaDReTGIB8QS-HVIpiXXkfn7ORAvUQc",
"AIzaSyBzJEBm6YcflpvhlN_U25ToSzqS_oQPCVc",
"AIzaSyDx_iJSwOU06YV6o_EVK37OlKwmmvMZbHM",
"AIzaSyCQdLWpZt_fmEPM7Nr0oRrXgUv2M2PvapQ",
],
[
"AIzaSyATTqIcxvmgXou_OIzRZy4mzzjOhBWUF08",
"AIzaSyAqnAi2jvfnVxUCMrQArvLnGs9Lh80KPds",
"AIzaSyChg0zNlTbCyc-FD_ZeZMWQ6bgMzXIgsk4",
"AIzaSyDFI-fykR8g-rKzDdyovhjFABm_vj3TUCw",
"AIzaSyCuruGb9LjUPYRvsTeVXcB-eJuGEQwSoFo",
"AIzaSyAsvlOX09ORB070ytFnNcMFBtpQJ2rFO88",
"AIzaSyBQtZ085RMv4VaDY82gvjRcnpYJYrJXpMk",
"AIzaSyCfBqHvfuW5cMKdSBfJ3CGOslOAE2b_9DY",
"AIzaSyCSnGNJsN0FeH_FX9qf2v5SG0Gdp_jm1ng",
"AIzaSyCvl6yTG7X7pYkIkdeiNmaoPk5uRtBSRk4",
"AIzaSyBDjbEee1dX1S9hrz5XJp1cFtNSqL_Ikgk",
"AIzaSyDsGFpoxQ9frUjSslSKda2j4-U0JO-FlYU",
"AIzaSyB5_pDd8tcXp3bUcdJQm9pAg9jfLShZ-lY",
"AIzaSyC1_w24CojGJhPe9R5xbdIwBbEpfyLKvaY",
"AIzaSyD8zyaoVptDwkSIT4UlbVDl2H0ZP-XYzx4",
"AIzaSyAdUujPDRX6klP9kK2rBVPeO4ybJAKDUpM",
],
[
"AIzaSyAjwa6Ril10ga9s91BydnNR6Tch25M_SDE",
"AIzaSyCiY5mViXhbYL1NwVjlsRIcSqxCb1etFvw",
"AIzaSyDntpdcZMe8C_G-_FXwSse-SL6I3O5daOI",
"AIzaSyC0a2j731Ul-K5OumRVsI0CFNa0SBabX88",
"AIzaSyBlNSvESj3SDYRg2NVEJfiicIiwvy2QR4A",
"AIzaSyB7XOdYxp44nu4roQb2KB6-Sw2bNV1bV6I",
"AIzaSyC1Td68jEhCQmKRKcRLHapyGsJ3l0uHjUI",
"AIzaSyAu4ERvCe8JgpJmaE4CY63DZZ66aXmHiNA",
"AIzaSyB6ojQUnu90wz5nGVxuF36TJYxQZbuK7f0",
"AIzaSyD5g6fyUWgKqxa-ZADdtSxvVKPp0YN5PYc",
"AIzaSyABXnfdBzivbdfAkctJZGRObtQLQF24C6Q",
"AIzaSyCyba29FRITwLVCYgFGtdthBuks1ssgCis",
"AIzaSyArTelfGiVShmhMNsYK0izS8d9QehT0BM8",
"AIzaSyAhUiTxhLUsF-w47T-OI4RTwXhzOTBOQK8",
"AIzaSyBQiuOdzpM40MLcwlM67oBN__ZQHyhIGEk",
"AIzaSyD_KCi_YM_X8Yk_HhQdKdQ65wyOVXvSIIs",
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
        



        
def event_prompt_structure(behavioral_description):
    
    event_prompt = f"""
                    {CONTEXT}
                    
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
                    - Order the events sequentially
                    - Substitute pronouns by the nouns they refer to.
                    - Substitue acronyms for their full description

                    Please generate the response in the form of a Python dictionary string with keys
                    "events". The value of "events" is a List(str), of which each item is an event.
                    DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python
                    dictionary string. For example, your response should look like this: {{"events":
                    [event1, event2, ...]}}. DO NOT INCLUDE ANY ADDITIONAL PUNCTUATION."""
                    
    return event_prompt
     

def entailment_prompt_structure(behavioral_description, events_json_string):
    entailment_prompt = f"""
                        Given a behavioral description of a patient and a list of events. For each event, classify the
                        relationship between the behavioral description and the event into three classes:
                        entailment, neutral, contradiction.
                        - "entailment" means that the video description entails the event.
                        - "contradiction" means that some detail in the video description contradicts
                        with the event.
                        - "neutral" means that the relationship is neither "entailment" or
                        "contradiction".
                        Output a list in Json format:
                        [ {{"event": "copy an event here", "relationship": "put class name here",
                        "reason": "give a reason"}}, ... ]
                        Behavioral description:
                        {behavioral_description}
                        Events:
                        {events_json_string}
                        DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the JSON.
                        Output:
                        """
                        
    return entailment_prompt

        
def get_events_from_behavioral_description(behavioral_description=None):
    
    if behavioral_description is None:
        raise ValueError("Please provide correct input...")
    
    # Events from behavioral description
    event_prompt = event_prompt_structure(behavioral_description)
    print(event_prompt)
    
    out = run_gemini(
        prompt=event_prompt,
        model_name="gemini-2.5-pro", # change to LLM (search which gemini model this is)
        temperature=0
    )
    
    events_json_string = out.text.strip()
    print(events_json_string)
    
    return events_json_string
    
    
def get_entailment_from_events(behavioral_description=None, events=None):
    
    if behavioral_description is None or events is None:
        raise ValueError("Please provide correct input...")
    
    # Events from behavioral description
    entailment_prompt = entailment_prompt_structure(behavioral_description=behavioral_description, events_json_string=events)
    print(entailment_prompt)
    out = run_gemini(
        prompt=entailment_prompt,
        model_name="gemini-2.5-pro",
        temperature=0
    )
    
    entailment_json_string = out.text.strip()
    print(entailment_json_string)

    return entailment_json_string
     
def calculate_entailment_score(data: List[Dict[str, str]]) -> float:
    """
    Calculate entailment score as (# entailment / # total)
    
    Args:
        data: List of dictionaries containing event information with 'relationship' field
        
    Returns:
        float: Entailment score between 0 and 1
    """
    
    if not data:
        return 0.0
    
    total_events = len(data)
    entailment_count = sum(1 for item in data if item.get('relationship', '').lower() == 'entailment')
    
    return entailment_count / total_events

def json_string_to_json_obj(json_string=""):
    json_str = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())
    parsed = json.loads(json_str)
    return parsed

                
def auto_dq_eval_metric(doctor_behavioral_description=None, VLM_behavioral_description=None):
    # Events from refrerence doctor description (Dref)
    Dref_events = get_events_from_behavioral_description(behavioral_description=doctor_behavioral_description)  
    Dref_events_in_Dmodel_json_string = get_entailment_from_events(behavioral_description=VLM_behavioral_description, events=Dref_events) 
    Dref_events_in_Dmodel_json = json_string_to_json_obj(Dref_events_in_Dmodel_json_string)
    
    # Events from candidate VLM description (Dmodel)
    Dmodel_events = get_events_from_behavioral_description(behavioral_description=VLM_behavioral_description)  
    Dmodel_events_in_Dref_json_string = get_entailment_from_events(behavioral_description=doctor_behavioral_description, events=Dmodel_events) 
    Dmodel_events_in_Dref_json = json_string_to_json_obj(Dmodel_events_in_Dref_json_string)

    # Precision and Recall
    precision = calculate_entailment_score(data=Dmodel_events_in_Dref_json)
    recall = calculate_entailment_score(data=Dref_events_in_Dmodel_json)

    return precision, recall
   
        
def main():
    
    seizure_behavioral_description =  """
                                    Patient is on the bed and suddenly feels an aura. 
                                    She extends her right arm and turns her head to the 
                                    right - non versive or forced. She then has gyratory 
                                    hand movements and then turns over to the right side 
                                    and around with eye deviation to the right. There are 
                                    complex body movements where she ends up moving in the bed 
                                    and climbs up. Many nurses and family members are holding her
                                    down, hence exact semiology is obscured but clearly there are 
                                    hypermotoric movements."""
                                    
    # Events from descriptiom
    events_json_string = get_events_from_behavioral_description(behavioral_description=seizure_behavioral_description)  

    # Entailments from events
    entailment_json_string = get_entailment_from_events(behavioral_description=seizure_behavioral_description, events=events_json_string) 
    entailment_json = json_string_to_json_obj(entailment_json_string)
    print(calculate_entailment_score(entailment_json))
           
    ####### I THINK entailment within itself is important too so that the model doesn't hallucinate if it does -> reprompting with the problematic entailments... #######
    # Output -- got entailment on its own of 10/10 --> later this will be Dmodel entailed in Dref and Dref entailed in Dmodel
               
    # # LLM Output
    # out = auto_dq_eval_metric(
    #     doctor=structure
    # )
    
    
    
    
if __name__ == "__main__":
    # main()
    print(run_gemini(prompt="give me ideas for the best dinner to eat?"))