from data_loaders import doctor_data_to_df, behavioral_descriptions_from_df, behavioral_df_to_list
from auto_dq_eval_metric import run_gemini, auto_dq_eval_metric, event_prompt_structure



def task3_prompt_structure(behavioral_description=""):
    
    ### EXAMPLE PROMPT ###
    example_prompt = f"""
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
    
    # TODO
    task3_prompt = """
                        TODO TODO TODO
                   """
                   
    return example_prompt

def main():
    # Load in the doctor data
    path_to_doctor_csv = 'data/Seizure-Data.xlsx'
    
    doctor_df = doctor_data_to_df(data_path=path_to_doctor_csv)
    # TODO: You might want to modify this function to index by file name instead of patient 
    # Current: A0002: "<behavioral_description>" -> A0002@5-13-2021@UA6693LK: "<behavioral_description>"
    behavioral_df = behavioral_descriptions_from_df(df=doctor_df)    
    
    
    # Connecting with the VLM to get "events"
    for patient, behavioral_description in behavioral_df.iterrows():
        
        # doctor description
        behavioral_description = str(behavioral_description['Behavioral description'])
        
        print(patient, behavioral_description)
        
            
        prompt = task3_prompt_structure(behavioral_description=behavioral_description)
        print(prompt)
        out = run_gemini(
            prompt=prompt,
            model_name="gemini-2.5-pro", # change to LLM (search which gemini model this is)
            temperature=0
        )
        
        text_output = out.text.strip()

        print("\nText Output:\n")
        print(text_output)
        

        # TODO -- remove this break to continue doing the rest of the behvioral descriptions from the patients
        break
    
    
if __name__ == "__main__":
    main()
    