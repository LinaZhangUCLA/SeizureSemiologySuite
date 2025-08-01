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
        
        
        
def auto_dq_eval_metric(context, reference=None, candidate=None):
    
    prompt = f"{context}\n\n{reference}"
    
    print(prompt)
    print("\n")
    
    out = run_gemini(
        prompt=prompt,
        model_name="gemini-2.5-pro",
        temperature=0
    )
    
    print(out)
    
        
        
        
        
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
    
    structure = f"""
                    Below is a description of a seizure event:
    
                    {seizure_event_description}
                    
                    Extract all key events from the above doctor description of a patient's seizure.
                    Requirements:
                    - An event must include a symptom
                    - Every event is represented by a brief sentence with in 10 words, with a
                    subject, a predicate and optionally an object, avoid unnecessary appearance
                    descriptions.
                    - Every event must be atomic, meaning that it cannot be further split into
                    multiple events.
                    - Substitute pronouns by the nouns they refer to.

                    Please generate the response in the form of a Python dictionary string with keys
                    "events". The value of "events" is a List(str), of which each item is an event.
                    DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python
                    dictionary string. For example, your response should look like this: {{"events":
                    [event1, event2, ...]}}"""
                    
    print(structure)
                    
    context = f"You are a seizure semiology expert analyzing doctors seizure reports. Please follow the instructions precisely."
    
    auto_dq_eval_metric(
        context=context,
        reference=structure
    )
    
    
if __name__ == "__main__":
    main()