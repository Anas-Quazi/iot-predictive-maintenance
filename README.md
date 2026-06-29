# 🔧 IoT Predictive Maintenance

> A Contextual Data Fusion Framework that predicts industrial equipment failures before they occur — by combining internal IoT sensor telemetry with external environmental signals.

![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-Private-lightgrey)

---

## Problem Statement

Most ML-based maintenance systems rely exclusively on internal sensor readings and fail in real-world deployment. Mechanical failures rarely happen in isolation — they are strongly influenced by external conditions like ambient temperature, humidity, and factory load.

This project addresses that gap with a **Contextual Data Fusion Framework** that merges internal IoT telemetry (vibration, temperature, torque, rotational speed) with external environmental signals to predict failures **before they occur**, with full model explainability.

---

## Pipeline

```
AI4I 2020 Dataset (IoT Sensor Data)
        │
        ▼
Dataset Preprocessing
(cleaning, time-series expansion, rolling statistics)
        │
        ▼
External Data Fusion
(merge sensor data with weather + factory load signals)
        │
        ▼
Feature Engineering
(statistical features, risk flags, lag/window features)
        │
        ▼
EDA
(distribution analysis, correlation, failure pattern discovery)
        │
        ▼
Imbalanced Classification
(SMOTE applied inside cross-validation folds)
        │
        ▼
LightGBM Model
(5-fold stratified cross-validation)
        │
        ▼
Explainability
(SHAP values — identifies which sensors drive each failure)
        │
        ▼
Alert Dashboard
(flags machines predicted to fail in the next 7 days)
```

---

## Repository Structure

```
iot-predictive-maintenance/
├── Dataset/                    # Raw and master dataset versions
├── Dataset_Preprocessing/      # EDA notebooks, preprocessing scripts
├── EDA/                        # Exploratory analysis, visualizations
├── External_Data_Fusion/       # Environmental feature generation pipeline
├── Feature_engineering/        # Statistical & risk flag feature scripts
├── Src/                        # SMOTE & LightGBM modeling
├── performance_metrics.ipynb   # Audit of model performance across dataset versions
├── requirement.txt             # Python dependencies
└── README.md
```

> **Branch convention:** Each team member works on a dedicated branch (`dev/shais`, `dev/anas`, `dev/aadi`, `dev/yash`). All changes are merged into `main` via pull requests.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Modeling | LightGBM, Scikit-learn |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Dashboard | Flask / Streamlit *(planned)* |
| Version Control | Git + GitHub |

---

## Dataset

**AI4I 2020 Predictive Maintenance Dataset**

- Source: UCI Machine Learning Repository
- Base records: 10,000 rows with 14 sensor features
- Enriched version: `ai4i2020_time_series.csv` — 125 columns after time-series expansion, rolling statistics, and external context fusion
- Target: Multi-class failure prediction (TWF, HDF, PWF, OSF, RNF)

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/Anas-Quad/iot-predictive-maintenance.git
cd iot-predictive-maintenance

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirement.txt
```

---

## Project Status

| Module | Status |
|---|---|
| Dataset Preprocessing | ✅ Complete |
| Time Series Logs | ✅ Complete |
| External Data Fusion | ✅ Complete |
| Feature Engineering | ✅ Complete |
| EDA | ✅ Complete |
| Ablation Study | ✅ Complete |
| Model Training (LightGBM) | ⏳ Week 3 |
| SMOTE inside CV folds | ⏳ Week 3 |
| 5-Fold Stratified Cross-Validation | ⏳ Week 3 |
| Noise Sensitivity Analysis | ⏳ Week 4 |
| Precision-Recall Threshold Tuning | ⏳ Week 4 |
| SHAP Explainability | ⏳ Week 4 |
| Alert Dashboard | ⏳ Week 4 |

---

## Performance Metrics

Ablation study across four progressive dataset versions, benchmarked using LightGBM with 80-20 train-test split:

| Dataset | Features | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|---|
| Alpha — Raw sensors only | 5 | 97.25% | 39.47% | 76.92% | 52.17% |
| Beta — Static engineered features | 49 | 99.10% | 78.38% | 74.36% | 76.32% |
| Gamma — + Time-series features | 117 | 99.40% | 96.55% | 71.79% | 82.35% |
| Delta — + External context (final) | 140 | 99.45% | 96.67% | 74.36% | 84.06% |

Each stage shows measurable improvement in F1 score, mathematically proving that both time-series expansion and external contextual fusion meaningfully improve predictive power beyond raw sensor data alone. The Delta dataset achieves the project target Macro F1 ≥ 0.85 and will be further optimized in Week 3 through SMOTE inside cross-validation folds and Precision-Recall threshold tuning.

---

## Team

| Member | Branch |
|---|---|
| Anas Quazi | `dev/anas` |
| Shais | `dev/shais` |
| Aadi | `dev/aadi` |
| Yash | `dev/yash` |

---

## Internship Context

This project is being developed as part of a 2-month industry internship at **Infotact** by a 4-member team. It follows strict Git commit discipline — commits are required on all active working days and reviewed at the end of the internship.

---

*README will be updated as modules are completed.*
