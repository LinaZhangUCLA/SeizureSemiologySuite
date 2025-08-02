from data_loaders import doctor_data_to_df, behavioral_descriptions_from_df, behavioral_df_to_list
from auto_dq_eval_metric import run_gemini, auto_dq_eval_metric, event_prompt_structure





# Task Evaluations
def main():
    
    # Load in the doctor data
    path_to_doctor_csv = 'data/Seizure-Data.xlsx'
    
    doctor_df = doctor_data_to_df(data_path=path_to_doctor_csv)
    behavioral_df = behavioral_descriptions_from_df(df=doctor_df)
    
    # Load in the VLM data -> depends on output
    # ##### TODO #####

    behavioral_list = behavioral_df_to_list(behavioral_df)
    
    
    
    
    # Connecting with the VLM to get "events"
    for patient, behavioral_description in behavioral_df.iterrows():
        
        # doctor description
        behavioral_description = str(behavioral_description['Behavioral description'])
        
        print(patient, behavioral_description)
        
        # print("\nPROMPT STRUCTURE\n")
        
        prompt = event_prompt_structure(behavioral_description=behavioral_description)
        
        # print(prompt)
        
        out = auto_dq_eval_metric(doctor=prompt)
        doctor_raw = out.text.strip()
        
        print(doctor_raw)
        
        # Testing the 

        
        
        # # as jsongit stash
        # try:
            
        break
    
    
    
    
    
    
    
        

if __name__ == '__main__':
    main()