# 🔧 IoT Predictive Maintenance

---

## Problem Statement

Most existing ML maintenance systems rely only on internal sensor signals and fail in real-world deployment. A machine's failure is rarely isolated — it is heavily influenced by external factors like weather conditions and factory load.

This project builds a **Contextual Data Fusion Framework** that integrates internal IoT telemetry (vibration, temperature, current) with external environmental signals to predict mechanical failures **before they happen**.

---

## Pipeline

```
IoT Sensor Data (AI4I Dataset)
        ↓
Data Ingestion & Signal Processing
(rolling mean, std, variance over time windows)
        ↓
Contextual Data Fusion
(merge sensor data with external context — weather, load)
        ↓
Feature Engineering
        ↓
Imbalanced Classification
(SMOTE inside cross-validation folds)
        ↓
LightGBM Model
(5-fold stratified cross-validation)
        ↓
Explainability
(SHAP values — which sensor caused the failure)
        ↓
Alert Dashboard
(which machine will fail in next 7 days)
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Modeling | LightGBM, Scikit-learn |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Dashboard | Flask / Streamlit |
| Version Control | Git + GitHub |

---
