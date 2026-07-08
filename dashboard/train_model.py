"""
train_model.py
Rebuilds features by replaying the raw dataset through feature_engine.py
(the SAME code path used live), chronological split, SMOTE on train only,
trains LightGBM, saves model + SHAP background + metrics.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import json
import pickle
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score, confusion_matrix)
from imblearn.over_sampling import SMOTE

from feature_engine import MachineBuffer, build_feature_columns, to_vector

RAW_COL_MAP = {
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Type": "type",
    "ambient_temp": "ambient_temp", "ambient_humidity": "ambient_humidity",
    "atmospheric_pressure": "atmospheric_pressure",
    "grid_voltage_fluctuation": "grid_voltage_fluctuation",
    "factory_load_density": "factory_load_density",
    "operator_skill_proxy": "operator_skill_proxy",
    "particulate_matter_pm10": "particulate_matter_pm10",
    "ambient_vibration_noise": "ambient_vibration_noise",
}

print("Loading raw dataset...")
df = pd.read_csv("raw_dataset.csv")
df = df.sort_values("Timestamp").reset_index(drop=True)

# --- Replay every row through the SAME causal feature engine used live ---
# tool_cycle reset (0) marks a new tool install -> new buffer (matches how
# the live simulator treats a "virtual machine" / tool session).
print("Replaying dataset through feature_engine (this guarantees zero train/serve skew)...")
buffers = {}
rows = []
labels = []
sub_labels = []  # TWF/HDF/PWF/OSF/RNF kept ONLY as auxiliary labels, never as features
timestamps = []

for i, row in df.iterrows():
    cycle_key = row["tool_cycle"]
    if cycle_key == 0:
        buf_id = f"tool_session_{i}"
        buffers[i] = MachineBuffer(buf_id)
        current_buf = buffers[i]
    reading = {v: row[k] for k, v in RAW_COL_MAP.items()}
    reading["type"] = int(row["Type"])
    feats = current_buf.push(reading)
    rows.append(feats)
    labels.append(int(row["Machine failure"]))
    sub_labels.append({k: int(row[k]) for k in ["TWF", "HDF", "PWF", "OSF", "RNF"]})
    timestamps.append(row["Timestamp"])

FEATURE_COLUMNS = build_feature_columns(rows[0])
X = pd.DataFrame([to_vector(r, FEATURE_COLUMNS) for r in rows], columns=FEATURE_COLUMNS)
y = pd.Series(labels)

print(f"Built {X.shape[0]} rows x {X.shape[1]} features. Failure rate: {y.mean():.4f}")
print("Feature columns:", FEATURE_COLUMNS)

# --- Chronological split (NOT random) — data is a time series ---
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
print(f"Train: {len(X_train)} rows ({y_train.sum()} failures) | "
      f"Test: {len(X_test)} rows ({y_test.sum()} failures)")

# --- SMOTE on TRAIN fold only ---
sm = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"After SMOTE (train only): {len(X_train_res)} rows, "
      f"failure rate {y_train_res.mean():.4f}")

# --- Train ---
model = lgb.LGBMClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    num_leaves=31, random_state=42, verbosity=-1,
)
model.fit(X_train_res, y_train_res)

# --- Evaluate on untouched, chronologically-later test set ---
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "test_failure_rate": float(y_test.mean()),
    "n_train": len(X_train_res),
    "n_test": len(X_test),
}
print(json.dumps(metrics, indent=2))

# sanity check: is it just predicting the majority class (always green)?
print("Prediction distribution on test set:", np.bincount(y_pred))
print("Probability range on test set:", y_proba.min(), y_proba.max(), y_proba.mean())

# --- Extra insights for the Global Insights dashboard tab ---
from sklearn.metrics import roc_curve, precision_recall_curve

fpr, tpr, roc_thresh = roc_curve(y_test, y_proba)
prec, rec, pr_thresh = precision_recall_curve(y_test, y_proba)

# subsample curve points so the frontend chart isn't overloaded
def subsample(arr, n=60):
    arr = list(arr)
    if len(arr) <= n:
        return arr
    idx = np.linspace(0, len(arr) - 1, n).astype(int)
    return [arr[i] for i in idx]

roc_curve_data = {"fpr": subsample(fpr), "tpr": subsample(tpr)}
pr_curve_data = {"precision": subsample(prec), "recall": subsample(rec)}

# probability histogram (shows the model spreads risk, not a single spike)
hist_counts, hist_edges = np.histogram(y_proba, bins=20, range=(0, 1))
prob_histogram = {"counts": hist_counts.tolist(), "edges": hist_edges.tolist()}

# failure-type breakdown across the WHOLE dataset (descriptive only — these
# columns are never used as model features, just reported here as context)
sub_df = pd.DataFrame(sub_labels)
failure_type_counts = {c: int(sub_df[c].sum()) for c in sub_df.columns}

# environmental correlation with predicted probability (test set only)
env_fields = ["ambient_temp", "ambient_humidity", "atmospheric_pressure",
              "grid_voltage_fluctuation", "factory_load_density",
              "operator_skill_proxy", "particulate_matter_pm10", "ambient_vibration_noise"]
env_test = df.iloc[split_idx:][env_fields].reset_index(drop=True)
env_correlation = []
for f in env_fields:
    if env_test[f].std() > 1e-9:
        corr = float(np.corrcoef(env_test[f].values, y_proba)[0, 1])
    else:
        corr = 0.0
    env_correlation.append({"field": f, "correlation": corr})
env_correlation.sort(key=lambda d: abs(d["correlation"]), reverse=True)

insights = {
    "roc_curve": roc_curve_data,
    "pr_curve": pr_curve_data,
    "probability_histogram": prob_histogram,
    "failure_type_counts": failure_type_counts,
    "env_correlation": env_correlation,
}
with open("model/insights.json", "w") as f:
    json.dump(insights, f, indent=2)
print("Saved model/insights.json")

# --- Normal operating ranges (5th-95th percentile) for the manual-input
# "how abnormal is this value" comparison chart ---
raw_field_map = {
    "air_temp": "Air temperature [K]", "process_temp": "Process temperature [K]",
    "rpm": "Rotational speed [rpm]", "torque": "Torque [Nm]", "tool_wear": "Tool wear [min]",
}
normal_ranges = {}
for feat, col in raw_field_map.items():
    normal_ranges[feat] = {
        "p5": float(df[col].quantile(0.05)),
        "p95": float(df[col].quantile(0.95)),
        "min": float(df[col].min()),
        "max": float(df[col].max()),
    }
with open("model/normal_ranges.json", "w") as f:
    json.dump(normal_ranges, f, indent=2)
print("Saved model/normal_ranges.json")


# --- SHAP explainer (TreeExplainer, background = train sample) ---
explainer = shap.TreeExplainer(model)

# --- Save everything ---
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model/explainer.pkl", "wb") as f:
    pickle.dump(explainer, f)
with open("model/feature_columns.json", "w") as f:
    json.dump(FEATURE_COLUMNS, f)
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Save a slice of historical (raw + sub-labels + timestamp) data for the
# "load from dataset" and "timestamp lookup" input modes, and for the
# simulator's replay backbone.
replay_df = df.copy()
replay_df["Machine_failure_label"] = y.values
for c in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
    replay_df[f"{c}_label"] = df[c]
replay_df.to_csv("model/replay_source.csv", index=False)

print("Saved model, explainer, feature_columns, metrics, replay_source.")
