import os
import random
import re
import json

content= """<think>
To determine the exact timestamp when the rhythmic jerking movements first appear, I'll look at the initial movements of the patient's arm and legs as signs of this phase of the seizure.
</think>
<answer>
{"timestamp": "00:11.00"}
</answer>"""


# llm_answer = content
# match = re.search(r"<answer>(.*?)</answer>", llm_answer)

# if match:
#     print("have <answer> tag")
#     llm_answer = match.group(1)
# else:
#     print(f"no <answer> tag")
#     llm_answer = content


# print(llm_answer)




llm_answer = """{
  "answer": "NA",
  "justification": "The video depicts a patient in a medical setting but does not show any interaction where the patient is verbally addressed by another person."
} """ 

gt_answer = """ {'answer': 'no', 'justification': 'A healthcare provider asks the patient, "You okay?" before the seizure, but the patient does not respond. During the seizure, the patient exhibits ictal vocalizations (moaning, crying) and tonic-clonic movements, and does not respond to verbal cues or demonstrate comprehension.'} """

llm_answer = json.loads(llm_answer)

print(llm_answer)

gt_answer = json.loads(gt_answer)


print(gt_answer)