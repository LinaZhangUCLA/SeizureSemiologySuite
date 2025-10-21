from openai import AsyncOpenAI 
from finetune import score_video_features

api_key = "<DASHSCOPE_API_KEY>"
model   = "qwen-plus"
video   = "video_001"

items = [
    {
      "feature":  "close_eyes",
      "gt_yesno": "yes", #小写
      "gt_text":  "the patient squeezes eyes tightly for several seconds...",
      "vlm_yesno":"yes", #小写
      "vlm_text": "eyes appear forcefully closed with sustained squeezing..."
    },
    {
      "feature":  "head_turning",
      "gt_yesno": "no", #小写
      "gt_text":  "no sustained head turning to either side was noted.",
      "vlm_yesno":"no", #小写
      "vlm_text": "head orientation remains mostly midline, no forced turn."
    },
    # ... 也可以继续加更多 feature
]

score = score_video_features(api_key, model, video, items)
print("final score =", score)  # 两位小数,一个float,所有 feature 得分的平均



############# Multi Videos##############
videos = {
    "video_001": [...同上 items...],
    "video_002": [...],
}
results = {}
for vid, items in videos.items():
    results[vid] = score_video_features(API_KEY, "qwen-plus", vid, items,
                                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
print(results)  # {video_001: 0.78, video_002: 0.65, ...}
