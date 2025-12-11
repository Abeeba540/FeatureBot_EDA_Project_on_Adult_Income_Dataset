# Reproducibility Report

## Random Seeds
- NumPy: 42 ✅
- Python: 42 ✅
- Stratified splits: 42 ✅
- K-Fold: 42 ✅
- Model: 42 ✅

## Dataset
- File: `data/adult.csv`
- Rows: 48,842
- Target: `income` mapped to 0 (<=50K), 1 (>50K)

## Results Consistency (Train/Val/Test split + Pipeline)
- Run 1: Test AUC 0.9075, F1 0.6802 ✅
- Run 2: Test AUC 0.9075, F1 0.6802 ✅
- Run 3: Test AUC 0.9075, F1 0.6802 ✅

Status: ✅ REPRODUCIBLE (metrics remain identical across runs with RANDOM_STATE=42)
