
import pandas as pd



def doctor_data_to_df(data_path):
    """
    Reads a CSV file containing doctor data and returns it as a pandas DataFrame.
    """
    df = pd.read_excel(data_path)
    return df
    
    
def behavioral_descriptions_from_df(df):
    # Get the patients and their behavioral descriptions
    patient_and_description = df[['Patient', 'Behavioral description']]
    patient_and_description.head()
    
    # Index by the patient name
    patient_and_description = patient_and_description.set_index('Patient')
    
    # Patients and their behavioral descriptions
    print(patient_and_description)
    
    
if __name__ == "__main__":
    # Example usage
    data_path = 'data/Seizure-Data.xlsx'  # Replace with your actual data path
    df = doctor_data_to_df(data_path)
    behavioral_descriptions_from_df(df)