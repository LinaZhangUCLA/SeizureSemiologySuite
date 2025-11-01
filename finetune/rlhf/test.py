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

<think>
The video shows a patient undergoing a medical examination or treatment. The patient appears to be experiencing distress or discomfort. A healthcare professional is interacting with the patient, possibly adjusting equipment or attempting to soothe the patient. The patient's facial expressions and body language suggest distress, which could be indicative of an epileptic seizure. However, without specific seizure symptoms like loss of consciousness, convulsions, or abnormal movements, it is not conclusive. Additional context such as background information or knowledge of the patient's medical history would be necessary for an accurate diagnosis.
</think>
{
  "answer": "NES",
  "description": "The patient appears to be experiencing distress or discomfort. A healthcare professional is interacting with the patient, possibly adjusting equipment or attempting to soothe the patient. The patient's facial expressions and body language suggest distress, which could be indicative of an epileptic seizure. However, without specific seizure symptoms like loss of consciousness, convulsions, or abnormal movements, it is not conclusive. Additional context such as background information or knowledge of the patient's medical history would be necessary for an accurate diagnosis."
}, gt answer {'answer': 'ES', 'description': 'The patient is laying in bed and his mom is giving him a massage at 14:15:32, he exhibits oroalimentary automatisms at 14:15:34, he exhibits dyscognitive language symptoms at 14:15:50. Right manual gestural automatisms are evident at 14:16:06, and bimanual gestural automatisms occur at 14:16:11, he looks to the left at 14:16:14, and the seizure terminates at 14:18:03.'}


llm_answer = """{
  "answer": "NA",
  "justification": "The video depicts a patient in a medical setting but does not show any interaction where the patient is verbally addressed by another person."
} """ 

gt_answer = """ {'answer': 'no', 'justification': 'A healthcare provider asks the patient, "You okay?" before the seizure, but the patient does not respond. During the seizure, the patient exhibits ictal vocalizations (moaning, crying) and tonic-clonic movements, and does not respond to verbal cues or demonstrate comprehension.'} """

llm_answer = json.loads(llm_answer)

print(llm_answer)

gt_answer = json.loads(gt_answer)


print(gt_answer)










