# FeatureBot: Adult Income Prediction with Feature Engineering

![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/ML-scikit--learn-orange)
![Reproducibility](https://img.shields.io/badge/Reproducibility-Verified-brightgreen)

## 🎯 Overview

**FeatureBot** is a production-ready machine learning project that predicts income levels (>$50K or ≤$50K) using the UCI Adult Census dataset. The project demonstrates **enterprise-grade MLOps practices** including:

- ✅ **Reproducible results** (identical metrics across runs via `RANDOM_STATE=42`)
- ✅ **Zero data leakage** (preprocessors fitted on train only)
- ✅ **Fairness-aware feature engineering** (excluded high-risk proxies)
- ✅ **Comprehensive documentation** (5 tracking documents + audit trail)
- ✅ **Automated CI/CD pipeline** (GitHub Actions verification)

### Performance

| Metric | Value |
|--------|-------|
| **Test AUC** | 0.9075 |
| **Test F1** | 0.6802 |
| **Precision** | ~0.57 |
| **Recall** | ~0.84 |
| **Reproducibility** | ✅ Verified (3+ runs) |

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/Abeeba540/FeatureBot_EDA_Project_on_Adult_Income_Dataset.git
cd FeatureBot_EDA_Project_on_Adult_Income_Dataset

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Training

```bash
python train_model.py
```

Expected output:
```
Global seeds set to 42
Data loaded: (48842, 15)
Target distribution BEFORE splitting:
  Class 0: 37155 (76.1%)
  Class 1: 11687 (23.9%)

Results (reproducible):
  Train AUC: 0.9071, F1: 0.6812
  Val AUC: 0.9089, F1: 0.6805
  Test AUC: 0.9075, F1: 0.6802

✅ Results saved. Run script again - results should be IDENTICAL!
```

### 3. Verify Reproducibility

```bash
python train_model.py  # Run 1
python train_model.py  # Run 2
# Both should show identical Test AUC: 0.9075, F1: 0.6802
```

### 4. View CI/CD Status

**GitHub Actions:** https://github.com/Abeeba540/FeatureBot_EDA_Project_on_Adult_Income_Dataset/actions

Latest run should show: **✅ Reproducibility Tests #PASSED**

---

## 📊 Project Structure

```
FeatureBot_EDA_Project_on_Adult_Income_Dataset/
├── README.md                           # This file
├── PROJECT_SUMMARY.md                  # Comprehensive project summary
├── train_model.py                      # Reproducible training script
├── requirements.txt                    # Dependencies
│
├── notebooks/
│   ├── FeatureBot_EDA_Project_on_Adult_Income_Dataset.ipynb
│   └── Tracking_Documents.ipynb
│
├── data/
│   └── adult.csv                       # UCI Adult dataset (48,842 rows)
│
├── outputs/
│   ├── feature_registry.csv            # Feature catalog (9 features)
│   ├── experiment_log.csv              # Experiment history
│   ├── experiment_metadata.json        # Detailed configs
│   ├── feature_justification.md        # Engineering rationale
│   ├── audit_trail.csv                 # Decision history
│   └── reproducibility_report.md       # Phase 2 verification
│
├── artifacts/
│   ├── cv_fold_indices.pkl
│   └── reproducibility_check_run1.pkl
│
├── .github/
│   └── workflows/
│       └── tests.yml                   # GitHub Actions CI/CD
│
└── docs/
    ├── ML_BEST_PRACTICES_COMPLETE_GUIDE.md
    ├── PRODUCTION_DEPLOYMENT_STEPS.md
    └── NEXT_STEPS_COMPLETE_ROADMAP.md
```

---

## 🔧 Technologies

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10 |
| **Data** | pandas 2.3.3, numpy 2.2.6 |
| **ML** | scikit-learn 1.7.2 |
| **Model Serialization** | joblib 1.5.2 |
| **Version Control** | Git/GitHub |
| **CI/CD** | GitHub Actions |
| **Notebooks** | Jupyter |

---

## 📈 Feature Engineering

### 9 Deployed Features

| # | Feature | Type | Definition |
|---|---------|------|-----------|
| 1 | `age_education_interaction` | Numeric Interaction | age × educational_num |
| 2 | `capital_net` | Numeric Derived | capital_gain - capital_loss |
| 3 | `has_capital_gain` | Binary Indicator | 1 if capital_gain > 0 else 0 |
| 4 | `has_capital_loss` | Binary Indicator | 1 if capital_loss > 0 else 0 |
| 5 | `is_overtime` | Binary Indicator | 1 if hours_per_week > 40 else 0 |
| 6 | `education_bucket` | Categorical Grouped | HS / Some College / Bachelors / Advanced |
| 7 | `is_professional` | Binary Indicator | 1 if occupation in {professional, executive, etc} |
| 8 | `professional_overtime` | Binary Interaction | is_professional × is_overtime |
| 9 | `hours_bin` | Categorical Binned | Part-time / Full-time / Overtime / High-overtime |

### Excluded Features (Fairness)

- ❌ `is_married` – 1.92× TPR disparity (gender bias proxy) – **Excluded**
- ❌ `age_married_interaction` – Built on high-risk feature – **Excluded**

**Fairness Trade-off:** -0.56% F1 for major fairness improvement. Acceptable.

---

## 🔐 Reproducibility

### Verification Method

```
Run 1: Test AUC = 0.9075, F1 = 0.6802
Run 2: Test AUC = 0.9075, F1 = 0.6802
Run 3: Test AUC = 0.9075, F1 = 0.6802
Status: ✅ REPRODUCIBLE (identical across runs)
```

### Reproducibility Checklist

| Criterion | Status | Implementation |
|-----------|--------|-----------------|
| Global Seeds | ✅ | `np.random.seed(42), random.seed(42)` |
| Stratified Splits | ✅ | `stratify=y, random_state=42` |
| CV Strategy | ✅ | `StratifiedKFold(5, random_state=42)` |
| Model Config | ✅ | `LogisticRegression(random_state=42)` |
| Preprocessing | ✅ | Deterministic scalers/encoders |
| No Data Leakage | ✅ | Pipeline fits on train only |
| CI/CD Verification | ✅ | GitHub Actions automated tests |

---

## 📊 Dataset

- **Source:** [UCI Adult Census Dataset](https://archive.ics.uci.edu/dataset/2/adult)
- **Rows:** 48,842
- **Features:** 14 original + 9 engineered = 23 total
- **Target:** Binary income (`<=50K`: 0, `>50K`: 1)
- **Class Distribution:** 76.1% low-income, 23.9% high-income
- **Location:** `data/adult.csv`

---

## 📄 Documentation

### Phase-Specific Documents

| Phase | Document | Purpose |
|-------|----------|---------|
| **Phase 1** | `README.md` | Project overview & quick start |
| **Phase 2** | `outputs/reproducibility_report.md` | Reproducibility verification |
| **Phase 3** | `outputs/feature_registry.csv` | Feature catalog |
| **Phase 3** | `outputs/experiment_log.csv` | Experiment history |
| **Phase 3** | `outputs/experiment_metadata.json` | Detailed configurations |
| **Phase 3** | `outputs/feature_justification.md` | Engineering rationale |
| **Phase 3** | `outputs/audit_trail.csv` | Decision log |
| **Phase 4** | `.github/workflows/tests.yml` | CI/CD configuration |

### Comprehensive Summary

- **`PROJECT_SUMMARY.md`** – Complete project documentation with all phases, results, and architecture

---

## 🎓 Key Results

### Performance Improvement

| Cycle | AUC | F1 | Change vs. Baseline |
|-------|-----|----|--------------------|
| **Baseline** (14 features) | 0.9066 | 0.6571 | — |
| **Cycle 1** (+5 features) | 0.9099 | 0.6652 | +0.33% AUC, +0.07% F1 |
| **Cycle 2** (+4 features, final) | **0.9075** | **0.6802** | +0.09% AUC, **+2.31% F1** |

**Notes:**
- Cycle 2 used **test set** (final evaluation)
- F1 improvement of +2.31% is significant for imbalanced classification
- Fairness trade-off: -0.56% F1 from excluding high-risk features (acceptable)

### Model Behavior

- **High Recall (84%)** – Catches most high-income individuals
- **Moderate Precision (57%)** – Some false positives acceptable for screening
- **Low Overfitting** – Train/Val/Test metrics nearly identical
- **Robust AUC (0.9075)** – Strong ranking ability across thresholds

---

## ⚖️ Fairness Considerations

### Subgroup Disparities Detected

**Gender (Male vs. Female):**
- TPR Disparity: 9.4%
- Status: Monitored, acceptable range

**Race (White vs. Non-white):**
- TPR Disparity: 13.9%
- Status: Monitored

**Marital Status (Married vs. Single):**
- TPR Disparity: 30.7% ❌ **High risk**
- Mitigation: **Excluded** `is_married` feature
- Trade-off: -0.56% F1 (acceptable for fairness)

### Monitoring Plan

- Daily fairness metric computation (framework ready)
- Threshold alerts if disparities exceed 15%
- Quarterly fairness audit
- Stakeholder reporting (documented in audit trail)

---

## 🚦 CI/CD Pipeline

### GitHub Actions Workflow

**File:** `.github/workflows/tests.yml`

**Triggers:**
- Every push to `main` or `master`
- Every pull request
- Manual trigger available

**Tests:**
1. ✅ Checkout code
2. ✅ Set up Python 3.10
3. ✅ Install dependencies
4. ✅ Run reproducibility test (Run 1)
5. ✅ Run reproducibility test (Run 2)
6. ✅ Compare results
7. ✅ Report status

**Status:** https://github.com/Abeeba540/FeatureBot_EDA_Project_on_Adult_Income_Dataset/actions

---

## 📋 How to Use

### Training a New Model

```python
# Python script or Jupyter notebook
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
df = pd.read_csv('data/adult.csv')

# Run train_model.py (all preprocessing handled)
# Output: Test AUC, F1, and artifacts
```

### Checking Results Reproducibility

```bash
# Run 1
python train_model.py > run1.log

# Run 2
python train_model.py > run2.log

# Compare (both should show identical metrics)
```

### Exploring Features

Open `outputs/feature_registry.csv` for:
- Feature definitions
- Dependencies
- Fairness implications
- Rationale for inclusion

### Reviewing Decisions

See `outputs/audit_trail.csv` for:
- When each feature was added
- Fairness audit results
- Decision to exclude high-risk features
- Reproducibility verification

---

## 🔄 Workflow

```
Data (adult.csv)
    ↓
[Split: Train 60% / Val 20% / Test 20%]
    ↓
[Preprocessing Pipeline]
  ├─ Numeric: Impute (median) → Scale (StandardScaler)
  └─ Categorical: Impute (mode) → Encode (OneHotEncoder)
    ↓
[Feature Engineering]
  └─ 9 engineered features added
    ↓
[Model Training]
  └─ LogisticRegression(random_state=42, solver='lbfgs')
    ↓
[Evaluation]
  ├─ Train: AUC=0.9071, F1=0.6812
  ├─ Val:   AUC=0.9089, F1=0.6805
  └─ Test:  AUC=0.9075, F1=0.6802
    ↓
[Verification]
  └─ ✅ Reproducible (identical across 3+ runs)
```

---

## 📦 Dependencies

```
pandas==2.3.3
numpy==2.2.6
scikit-learn==1.7.2
joblib==1.5.2
```

Install via:
```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

**Note:** All PRs trigger GitHub Actions verification.

---

## 📧 Contact & Support

- **Author:** Ummu Abeeba
- **Email:** abeeba2430@gmail.com
- **GitHub:** [@Abeeba540](https://github.com/Abeeba540)
- **Issues:** [GitHub Issues](https://github.com/Abeeba540/FeatureBot_EDA_Project_on_Adult_Income_Dataset/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Abeeba540/FeatureBot_EDA_Project_on_Adult_Income_Dataset/discussions)

---

## 📚 References

- **Dataset:** [UCI Machine Learning Repository - Adult](https://archive.ics.uci.edu/dataset/2/adult)
- **Best Practices:** ML_BEST_PRACTICES_COMPLETE_GUIDE.md
- **Deployment:** PRODUCTION_DEPLOYMENT_STEPS.md
- **Roadmap:** NEXT_STEPS_COMPLETE_ROADMAP.md
- **Full Summary:** PROJECT_SUMMARY.md

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🏆 Project Status

| Phase | Status | Date |
|-------|--------|------|
| ✅ Phase 1: Repository Setup | Complete | Dec 8, 2025 |
| ✅ Phase 2: Reproducible Training | Complete | Dec 10, 2025 |
| ✅ Phase 3: Documentation & Tracking | Complete | Dec 12, 2025 |
| ✅ Phase 4: CI/CD Automation | Complete | Dec 12, 2025 |

**Overall Status:** ✅ **Production Ready**

---

**Last Updated:** December 12, 2025  
**Created by:** Ummu Abeeba  
**Repository:** https://github.com/Abeeba540/FeatureBot_EDA_Project_on_Adult_Income_Dataset
