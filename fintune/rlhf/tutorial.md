# 使用Qwen2.5-7B-omni模型进行GRPO训练

## step 1: 配置镜像

modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py311-torch2.6.0-1.29.0-LLM

链接：https://modelscope.cn/docs/intro/environment-setup#%E6%9C%80%E6%96%B0%E9%95%9C%E5%83%8F

optional: 也许需要单独安装一下math_verify: pip install math_verify==0.5.2

pip install rouge
pip install sacrebleu

## step 2: 替换文件

### 1.数据集预处理文件替换

`ms-swift-main/swift/llm/dataset/dataset/mllm.py`

数据集预处理，使得训练集符合ms-swift训练要求

### 2.系统prompt定义文件替换

`ms-swift-main/examples/train/grpo/prompt.txt`

seizure的grpo训练的系统命令，可以替换为指定seizure expert的角色

### 3.奖励函数计算文件替换

`ms-swift-main/examples/train/grpo/plugin/plugin.py`

引入了SerizureORM计算不同task的reward




## step 3: 一键启动脚本grpo.sh

```bash
bash grpo.sh
```