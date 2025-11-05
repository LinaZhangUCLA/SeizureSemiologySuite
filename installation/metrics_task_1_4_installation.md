## Environment Installation (for Metrics Calculation)

This environment is used for running evaluation metrics and data analysis scripts.

```bash
# 1. Create new conda environment
conda create -n metrics python=3.10 -y
conda activate metrics

# 2. Install required packages
pip install pandas>=2.0.0 scikit-learn>=1.3.0 numpy>=1.24.0
```

### Package Requirements
- Python 3.10
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0

pip install sacrebleu
pip install rouge-score
pip install bert-score

### Usage
After installation, activate the environment before running any metrics calculation scripts:
```bash
conda activate metrics
```
