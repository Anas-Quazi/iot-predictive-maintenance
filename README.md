# 🔧 IoT Predictive Maintenance

> A Contextual Data Fusion Framework that predicts industrial equipment failures before they occur — by combining internal IoT sensor telemetry with external environmental signals, and serves those predictions through a live, explainable monitoring dashboard.

![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-Private-lightgrey)

---

## Problem Statement

Most ML-based maintenance systems rely exclusively on internal sensor readings and fail in real-world deployment. Mechanical failures rarely happen in isolation — they are strongly influenced by external conditions like ambient temperature, humidity, and factory load.

This project addresses that gap with a **Contextual Data Fusion Framework** that merges internal IoT telemetry (vibration, temperature, torque, rotational speed) with external environmental signals to predict failures **before they occur**, with full model explainability — and ships that model behind a live dashboard that a maintenance team could actually use.

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
Feature Engineering
(statistical features, risk flags, lag/window features)
        │
        ▼
External Data Fusion
(merge sensor data with weather + factory load signals)
        │
        ▼
EDA
(distribution analysis, correlation, failure pattern discovery)
        │
        ▼
Imbalanced Classification
(SMOTE applied inside cross-validation folds / train-fold only)
        │
        ▼
LightGBM Model
(chronological split — no sub-label leakage, see "Model Integrity" below)
        │
        ▼
Explainability
(SHAP values — identifies which sensors drive each failure)
        │
        ▼
Live Dashboard
(Flask + Chart.js — fleet monitoring, manual what-if, timestamp
 lookup/range, live simulated IoT stream, global model insights)
```

---

## Repository Structure

```
iot-predictive-maintenance/
├── Dataset/                       # Raw and master dataset versions
├── Dataset_Preprocessing/         # Preprocessing scripts
├── EDA/                           # Exploratory analysis, visualizations
├── External_Data_Fusion/          # Environmental feature generation pipeline
├── Feature_engineering/           # Statistical & risk flag feature scripts
├── models/                        # Saved research-track model artifacts
├── src/                           # Core training/eval pipeline
│   ├── cv_engine.py
│   ├── data_prep.py
│   ├── evaluate_robustness.py
│   ├── model_seal.py
│   ├── stress_tester.py
│   ├── train_lgbm.py
│   └── tune_noise_threshold.py
├── reports/                       # All output artifacts (plots, metrics)
│   ├── figures/
│   │   └── noise_pr_curve.png
│   └── metrics/
│       └── performance_metrics.json
├── dashboard/                      # Live serving layer (Flask app)
│   ├── app.py                      # Routes, scoring, SHAP, all input modes
│   ├── feature_engine.py           # Causal feature pipeline (single source of truth)
│   ├── simulator.py                # Live "IoT stream" simulator, degradation model
│   ├── train_model.py              # Retrains the causal, leak-free serving model
│   ├── requirements.txt
│   ├── model/                      # Model, SHAP explainer, cached insights
│   ├── static/                     # CSS + JS (Chart.js dashboard)
│   └── templates/                  # index.html
├── .gitignore
├── requirement.txt                 # Root research-environment dependencies
└── README.md
```

> **Branch convention:** Each team member works on a dedicated branch (`dev/shais`, `dev/anas`, `dev/aadi`, `dev/yash`). All changes are merged into `main` via pull requests.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+, JavaScript (ES6) |
| Data Processing | Pandas, NumPy |
| Visualization (research) | Matplotlib, Seaborn |
| Visualization (dashboard) | Chart.js |
| ML Modeling | LightGBM, Scikit-learn |
| Imbalance Handling | imbalanced-learn (SMOTE, train-fold only) |
| Explainability | SHAP (TreeExplainer) |
| Dashboard Backend | Flask |
| Version Control | Git + GitHub |

---

## Dataset

**AI4I 2020 Predictive Maintenance Dataset**

- Source: UCI Machine Learning Repository
- Base records: 10,000 rows with 14 sensor features
- Enriched version: `predictive_maintenance_master_features.csv` — 140+ columns after time-series expansion, rolling statistics, and external context fusion
- Target: `Machine failure` (binary), with `TWF / HDF / PWF / OSF / RNF` as the five failure-mode sub-labels
- **Important structural note:** the enriched dataset is one continuous 7-day, 1-minute-interval stream — not multiple labeled machines. `tool_cycle` ramps once from 0→119 rather than resetting repeatedly. The dashboard's "fleet" is an honest partition of this single stream into equal time-block "virtual machines," documented as such rather than presented as real multi-machine telemetry.

---

## Model Integrity — What We Checked and Fixed

Before treating any accuracy number as real, we specifically tested for the three most common ways a predictive-maintenance model silently cheats:

| Risk | Finding | Fix |
|---|---|---|
| **Target leakage via sub-labels** | `Machine failure` matched `OR(TWF, HDF, PWF, OSF, RNF)` in 99.73% of rows — these columns are disaggregated components of the target, not independent predictors. | Excluded from all model features. Kept only as descriptive/auxiliary labels for the failure-type breakdown shown in the dashboard. |
| **Random split leakage on time-series data** | A random 80/20 split lets rolling/lag features "see" neighboring rows from the same local window during evaluation, inflating scores. | Chronological split — model is evaluated only on the last 20% of the timeline, which it never trained on. |
| **SMOTE applied before splitting** | Oversampling before the split leaks synthetic near-duplicates of test-set failures into training. | SMOTE fit and applied strictly on the training fold only. |
| **"Always green" / majority-class collapse** | Verified directly: probability output on the holdout ranges from 7.1×10⁻⁶ to 0.9999, mean 0.017 — a real spread, not a constant. | N/A — confirmed not an issue after the fixes above. |
| **Live simulation showing unrealistic risk levels** | Initial drift model pushed every virtual machine toward failure simultaneously, producing far more "Critical" alerts than the real ~3.4% failure rate would suggest. | Rebuilt as a per-machine, independently-triggered degradation episode (rare onset probability, random severity/duration, then a simulated maintenance reset). Verified offline over 300 ticks × 12 machines: **3.75% critical rate**, closely matching the dataset's real 3.4% failure prevalence, with only 0–1 machines elevated at any given moment. |

**Causal, zero-skew feature pipeline:** `dashboard/feature_engine.py` is the single implementation of every rolling/lag/z-score/physics-threshold feature, used identically by training, the live simulator, and manual "what-if" input. This guarantees the model is trained and served on exactly the same feature logic — a common, often-invisible source of production bugs.

---

## Performance Metrics

### Research track (ablation study, `src/`)

Ablation study across four progressive dataset versions, benchmarked using LightGBM with an 80–20 train-test split:

| Dataset | Features | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|---|
| Alpha — Raw sensors only | 5 | 97.25% | 39.47% | 76.92% | 52.17% |
| Beta — Static engineered features | 49 | 99.10% | 78.38% | 74.36% | 76.32% |
| Gamma — + Time-series features | 117 | 99.40% | 96.55% | 71.79% | 82.35% |
| Delta — + External context (final) | 140 | 99.45% | 96.67% | 74.36% | 84.06% |

Each stage shows measurable improvement in F1 score, demonstrating that both time-series expansion and external contextual fusion meaningfully improve predictive power beyond raw sensor data alone.

### Production / dashboard track (`dashboard/train_model.py`)

A separate model, trained specifically for live serving, using the causal feature pipeline described above and a **chronological** (not random) split:

| Metric | Value |
|---|---|
| Accuracy | 99.25% |
| Precision | 90.0% |
| Recall | 69.2% |
| F1 Score | 0.783 |
| ROC-AUC | 0.977 |
| Test set | Last 20% of the timeline, chronologically held out (2,000 rows, 39 real failures) |

**Why the two tracks differ:** the research-track ablation study and the dashboard's production model use different splitting methodology and different feature sets (the dashboard rebuilds a smaller, purely causal feature set so it can be computed identically in a live stream). Both are legitimate, but they answer different questions — the ablation study measures how much each feature group helps in principle; the production metrics measure what the deployed model will actually achieve on genuinely unseen future data. Neither includes the TWF/HDF/PWF/OSF/RNF sub-labels as inputs.

---

## Live Dashboard

The `dashboard/` app turns the trained model into something a maintenance operator could actually use.

### Input modes

1. **Manual / What-If** — enter raw sensor readings for a hypothetical machine. Optionally borrow real historical context from a chosen timestamp so rolling/lag features aren't undefined (a clearly labeled cold-start warning appears if you don't).
2. **Timestamp Lookup** — enter an exact historical timestamp; see the model's prediction against the actual recorded outcome.
3. **Timestamp Range** — enter a start/end timestamp; every row in that window is replayed through the causal pipeline and scored, returning aggregate risk, dominant failure driver, environment averages, and a recommendation for the whole window.
4. **Live simulated IoT stream** — a background thread continuously "ticks" 12 virtual machines (partitioned from the real historical stream) with a realistic, independently-triggered degradation model, so the fleet view updates like a live monitoring system.

### Output, per prediction

- **Failure probability** (not just a binary flag)
- **Risk tier** — Low / Medium / High / Critical
- **Primary failure-mode reason** — mapped from physics-based thresholds (Tool Wear / Heat Dissipation / Power / Overstrain Failure conditions, per the published AI4I synthetic failure rules)
- **Top SHAP drivers** — which features pushed this specific prediction up or down
- **Recommended action** — plain-language next step tied to the dominant driver

### Visualizations

- Fleet table with live risk tiers, trend arrows, and color-coded rows
- KPI bar (counts per risk tier)
- Failure-probability gauge and SHAP driver bars per machine
- Sensor trend charts and risk-over-time charts
- Manual input: entered values plotted against the dataset's normal 5th–95th percentile operating range
- Timestamp range: risk trajectory, aggregated SHAP drivers, environment averages
- Global Insights tab: feature importance, ROC curve, Precision-Recall curve, predicted-probability distribution, historical failure-type breakdown, environment-correlation chart

### Running it

```bash
cd dashboard
pip install -r requirements.txt
python app.py
# open http://localhost:5050
```

---

## Known Limitations & Honest Caveats

- **Single-stream fleet simulation**: the "12 virtual machines" are a partition of one real continuous sensor stream, not independently sourced machines. This is disclosed in the dashboard's Session Replay tab rather than presented as more than it is.
- **Environmental correlation is weak in this dataset**: measured correlation between environmental fields (ambient temp, humidity, factory load, etc.) and predicted risk came out at roughly 0.01–0.03 — present, but weak. The contextual-fusion story is architecturally real (the model does take these features as input) but its measured impact on this particular dataset is modest; worth stating plainly rather than overselling.
- **RNF (random failure) is under-supported**: only 19 positive examples in 10,000 rows. We rely on physics-threshold flags and overall model probability rather than a dedicated per-failure-type classifier for this category, since 19 examples isn't enough to model reliably on its own.
- **Recall on the held-out chronological test set is 69.2%**, not higher — meaning roughly 3 in 10 real failures in that window were missed at the current probability threshold. This is disclosed rather than rounded up, and the risk-tier thresholds can be tuned against the Precision-Recall curve in Global Insights depending on whether the deployment prioritizes catching more failures (lower threshold) or reducing false alarms (higher threshold).

---

## Future Work

- Replace the simulator with a real MQTT-based ingestion path (`POST /api/ingest`) so real sensors can be swapped in without touching the feature engine, model, or dashboard.
- Periodic retraining pipeline as more real operational data accumulates.
- Dedicated per-failure-type models if/when enough positive examples exist for each (especially RNF).
- Threshold tuning per deployment context using the Precision-Recall curve already exposed in Global Insights.

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
| Model Training (LightGBM) | ✅ Complete |
| SMOTE inside CV folds / train-fold only | ✅ Complete |
| Leakage audit (sub-labels, split methodology) | ✅ Complete |
| SHAP Explainability | ✅ Complete |
| Live Dashboard (Flask) | ✅ Complete |
| Manual / Timestamp / Range input modes | ✅ Complete |
| Live simulated IoT stream (calibrated) | ✅ Complete |
| Global Insights (ROC, PR, importance, env. correlation) | ✅ Complete |

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/Anas-Quad/iot-predictive-maintenance.git
cd iot-predictive-maintenance

# 2. Create and activate a virtual environment (research pipeline)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install research-track dependencies
pip install -r requirement.txt

# 4. To run the live dashboard instead
cd dashboard
python app.py
```

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

*README last updated after completion of the live dashboard, model integrity audit, and simulation recalibration.*
