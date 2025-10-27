# 使用Qwen2.5-7B-omni模型进行GRPO训练

## step 1: 配置镜像

modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py311-torch2.6.0-1.29.0-LLM

链接：https://modelscope.cn/docs/intro/environment-setup#%E6%9C%80%E6%96%B0%E9%95%9C%E5%83%8F


## step 2: 安装包

`
pip install math_verify==0.5.2
pip install rouge
pip install sacrebleu
`

## step 3: 替换文件


### 奖励函数计算文件替换


将`SeizureSemiologyBench/fintune/rlhf/plugin.py`的文件替换到`ms-swift-main/examples/train/grpo/plugin/plugin.py`

引入了SerizureORM计算不同task的reward



## step 4: 一键启动脚本grpo.sh

```bash
bash qwen2_5_omni_task_1_7_grpo.sh
```