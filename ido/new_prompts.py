new_prompts = [
    "You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Please assess whether the patient shows impaired consciousness or awareness during the event. First, explain your answer in 3 sentences within parentheses i.e. (\"answer\"). Then write a new line in the form \"\\n\". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.",
    
    "You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Please assess whether the patient displays altered mental state or disorientation during the seizure. First, explain your answer in 3 sentences within parentheses i.e. (\"answer\"). Then write a new line in the form \"\\n\". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.",
    
    "You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Please assess whether the patient shows automatisms or inappropriate behavioral responses to their surroundings. First, explain your answer in 3 sentences within parentheses i.e. (\"answer\"). Then write a new line in the form \"\\n\". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.",
    
    "You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Please assess whether the patient exhibits involuntary eyelid movements or forced eye positioning. First, explain your answer in 3 sentences within parentheses i.e. (\"answer\"). Then write a new line in the form \"\\n\". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.",
    
    "You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Please assess whether the patient shows myoclonic jerks specifically affecting the eyelids. First, explain your answer in 3 sentences within parentheses i.e. (\"answer\"). Then write a new line in the form \"\\n\". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.",
    
    "You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Please assess whether the patient displays involuntary eye movements in a repetitive pattern. First, explain your answer in 3 sentences within parentheses i.e. (\"answer\"). Then write a new line in the form \"\\n\". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.",
    
    "You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Please assess whether the patient exhibits excessive saliva production or foaming at the mouth. First, explain your answer in 3 sentences within parentheses i.e. (\"answer\"). Then write a new line in the form \"\\n\". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.",
    
    "You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Please assess whether the patient shows sudden myoclonic movements in any extremities or body regions. First, explain your answer in 3 sentences within parentheses i.e. (\"answer\"). Then write a new line in the form \"\\n\". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.",
    
    "You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Please assess whether the patient experiences atonic seizure activity with sudden collapse. This is seen through a loss of muscle tone and drop. First, explain your answer in 3 sentences within parentheses i.e. (\"answer\"). Then write a new line in the form \"\\n\". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.",
    
    "You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Please assess whether the patient shows violent motor activity characteristic of frontal lobe seizures. First, explain your answer in 3 sentences within parentheses i.e. (\"answer\"). Then write a new line in the form \"\\n\". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.",
    
    "You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Please assess whether the patient displays asymmetric tonic posturing of the upper extremities known as \"fencing\". First, explain your answer in 3 sentences within parentheses i.e. (\"answer\"). Then write a new line in the form \"\\n\". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.",
    
    "You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Please assess whether the patient shows severe extensor posturing with spinal hyperextension. First, explain your answer in 3 sentences within parentheses i.e. (\"answer\"). Then write a new line in the form \"\\n\". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.",
    
    "You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Please assess whether the patient shows spasm movements or brief tonic seizure activity in their trunks or limbs. First, explain your answer in 3 sentences within parentheses i.e. (\"answer\"). Then write a new line in the form \"\\n\". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.",
    
    "You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Does the patient exhibit autonomic changes (flushing, pallor, apnea) during the event? First, explain your answer in 3 sentences within parentheses i.e. (\"answer\"). Then write a new line in the form \"\\n\". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.",
    
    "You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Please assess whether the patient shows difficulty speaking or understanding language after the seizure. Specifically, if they exhibit ictal or postictal aphasia or language dysfunction. First, explain your answer in 3 sentences within parentheses i.e. (\"answer\"). Then write a new line in the form \"\\n\". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.",
    
    "You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Please assess whether the patient exhibits sleepiness, postictal confusion, or somnolence following the seizure. First, explain your answer in 3 sentences within parentheses i.e. (\"answer\"). Then write a new line in the form \"\\n\". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer.",
    
    "You are a neurologist doctor with specialization in seizure disorders. You are the best at identifying non-epileptic seizures from epileptic seizures. Please assess whether the patient shows weakness in the limbs such as Todd's paralysis or focal postictal neurological deficits. First, explain your answer in 3 sentences within parentheses i.e. (\"answer\"). Then write a new line in the form \"\\n\". Finally, Answer with 'yes' or 'no'. Only use one of the possible answers: 'yes' or 'no'. Do not include any extra text in your output—only the answer."
]

new_features = [
    "impaired_awareness",
    "dazed_state",
    "inappropriate_interaction",
    "forced_eye_open_or_close",
    "eyelid_myoclonia",
    "nystagmus",
    "hypersalivation",
    "myoclonic_jerk",
    "atonic_drop",
    "hypermotor_activity",
    "fencing_posture",
    "opisthotonus",
    "epileptic_spasm",
    "autonomic_change",
    "language_disturbance (aphasia)",
    "postictal_confusion",
    "todds_paralysis"
]

print(f"Total number of prompts: {len(new_prompts)}")
print(f"Total number of features: {len(new_features)}")

print("\nFirst prompt:")
print(new_prompts[0])
print("\nFirst feature:")
print(new_features[0])