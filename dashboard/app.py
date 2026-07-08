"""
app.py — Flask backend for the predictive maintenance dashboard.

Input modes (all funnel through feature_engine.MachineBuffer + the same model):
  1. /api/manual_predict      -> single manual raw-value entry (what-if), optional
                                  borrowed history from a timestamp
  2. /api/lookup_timestamp    -> exact historical row lookup + prediction
  3. /api/machine_range       -> replay a chosen tool_cycle session / time range
  4. background simulator     -> continuous live "IoT" stream (12 virtual machines)

Output per prediction: probability, risk tier, likely failure-mode driver
(from physics risk flags), top SHAP contributors, recommended action.
"""
import pickle
import json
import threading
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template

from feature_engine import MachineBuffer, to_vector, TYPE_MAP
from simulator import FleetState, Simulator

app = Flask(__name__)

# ---------- Load trained artifacts ----------
with open("model/model.pkl", "rb") as f:
    MODEL = pickle.load(f)
with open("model/explainer.pkl", "rb") as f:
    EXPLAINER = pickle.load(f)
with open("model/feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)
with open("model/metrics.json") as f:
    MODEL_METRICS = json.load(f)
with open("model/insights.json") as f:
    MODEL_INSIGHTS = json.load(f)
with open("model/normal_ranges.json") as f:
    NORMAL_RANGES = json.load(f)

REPLAY_DF = pd.read_csv("model/replay_source.csv").sort_values("Timestamp").reset_index(drop=True)

print("Precomputing features for the full historical dataset (one-time, used by "
      "timestamp lookup / range insights so those stay fast and consistent)...")
_buf = MachineBuffer("full_history")
FULL_FEATS = []
for _, r in REPLAY_DF.iterrows():
    reading = {
        "air_temp": r["Air temperature [K]"], "process_temp": r["Process temperature [K]"],
        "rpm": r["Rotational speed [rpm]"], "torque": r["Torque [Nm]"],
        "tool_wear": r["Tool wear [min]"], "type": int(r["Type"]),
        "ambient_temp": r.get("ambient_temp"), "ambient_humidity": r.get("ambient_humidity"),
        "atmospheric_pressure": r.get("atmospheric_pressure"),
        "grid_voltage_fluctuation": r.get("grid_voltage_fluctuation"),
        "factory_load_density": r.get("factory_load_density"),
        "operator_skill_proxy": r.get("operator_skill_proxy"),
        "particulate_matter_pm10": r.get("particulate_matter_pm10"),
        "ambient_vibration_noise": r.get("ambient_vibration_noise"),
    }
    FULL_FEATS.append(_buf.push(reading))
FULL_X = pd.DataFrame([to_vector(f, FEATURE_COLUMNS) for f in FULL_FEATS], columns=FEATURE_COLUMNS)
FULL_PROBA = MODEL.predict_proba(FULL_X)[:, 1]
print(f"Cached features + probabilities for {len(FULL_FEATS)} historical rows. "
      f"Prob range: {FULL_PROBA.min():.5f}-{FULL_PROBA.max():.5f}, mean {FULL_PROBA.mean():.5f}")

RISK_TIERS = [
    (0.05, "Low"),
    (0.20, "Medium"),
    (0.50, "High"),
    (1.01, "Critical"),
]

FAILURE_REASON_MAP = {
    "twf_risk_flag": "Tool Wear Failure risk — tool wear in critical replacement window",
    "hdf_risk_flag": "Heat Dissipation Failure risk — low temp differential + low rotational speed",
    "pwf_risk_flag": "Power Failure risk — power output outside safe operating band",
    "osf_risk_flag": "Overstrain Failure risk — torque x tool wear exceeds type-rated threshold",
}
ACTION_MAP = {
    "twf_risk_flag": "Schedule tool replacement before next shift.",
    "hdf_risk_flag": "Check coolant/heat dissipation system and rotational speed setpoint.",
    "pwf_risk_flag": "Inspect power supply/load — running outside rated power band.",
    "osf_risk_flag": "Reduce load or replace tool — cumulative strain has exceeded rated threshold.",
    "torque": "Inspect torque/load — trending abnormally high.",
    "tool_wear": "Tool wear trending up — plan replacement soon.",
    "process_temp": "Monitor cooling — process temperature trending high.",
    "power": "Inspect drive/power system — output trending abnormally.",
}


def tier_for(prob):
    for cutoff, name in RISK_TIERS:
        if prob < cutoff:
            return name
    return "Critical"


def score_features(feats: dict):
    """Given a full feature dict, return probability/tier/reasons/action/shap."""
    vec = to_vector(feats, FEATURE_COLUMNS)
    X = pd.DataFrame([vec], columns=FEATURE_COLUMNS)
    proba = float(MODEL.predict_proba(X)[0, 1])
    tier = tier_for(proba)

    shap_values = EXPLAINER.shap_values(X)
    sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    contrib = sorted(zip(FEATURE_COLUMNS, sv), key=lambda kv: abs(kv[1]), reverse=True)
    top_drivers = [{"feature": k, "impact": float(v)} for k, v in contrib[:5]]

    active_flags = [k for k in FAILURE_REASON_MAP if feats.get(k)]
    if active_flags:
        primary_reason = FAILURE_REASON_MAP[active_flags[0]]
        action = ACTION_MAP[active_flags[0]]
    elif top_drivers:
        top_feat = top_drivers[0]["feature"]
        base_feat = next((k for k in ACTION_MAP if k in top_feat), None)
        primary_reason = f"No physics threshold breached; model driven mainly by '{top_feat}'"
        action = ACTION_MAP.get(base_feat, "No immediate action — continue routine monitoring.")
    else:
        primary_reason = "Nominal operating conditions"
        action = "No action needed."

    return {
        "probability": proba,
        "risk_tier": tier,
        "primary_reason": primary_reason,
        "recommended_action": action,
        "top_drivers": top_drivers,
        "active_risk_flags": active_flags,
    }


# ---------- Live simulator wiring ----------
FLEET_STATE = FleetState()


def scorer_callback(machine_id, feats):
    result = score_features(feats)
    FLEET_STATE.record_risk(machine_id, pd.Timestamp.now().isoformat(), result["probability"])
    if result["risk_tier"] in ("High", "Critical"):
        FLEET_STATE.add_alert(machine_id, result["risk_tier"], result["probability"],
                               pd.Timestamp.now().isoformat())


SIMULATOR = Simulator(FLEET_STATE, scorer_callback=scorer_callback)
SIMULATOR.start()


# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/fleet_status")
def fleet_status():
    snap = FLEET_STATE.snapshot()
    machines = []
    for machine_id, feats in snap["latest_features"].items():
        result = score_features(feats)
        risk_hist = snap["risk_history"].get(machine_id, [])
        delta = 0.0
        if len(risk_hist) >= 2:
            delta = risk_hist[-1]["p"] - risk_hist[-2]["p"]
        machines.append({
            "machine_id": machine_id,
            "probability": result["probability"],
            "risk_tier": result["risk_tier"],
            "primary_reason": result["primary_reason"],
            "delta": delta,
            "tool_wear": feats.get("tool_wear"),
            "process_temp": feats.get("process_temp"),
        })
    machines.sort(key=lambda m: m["probability"], reverse=True)

    tiers_count = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
    for m in machines:
        tiers_count[m["risk_tier"]] += 1

    return jsonify({
        "machines": machines,
        "kpis": {
            "total_machines": len(machines),
            "tier_counts": tiers_count,
        },
        "alerts": snap["alerts"][:10],
    })


@app.route("/api/machine/<machine_id>")
def machine_detail(machine_id):
    snap = FLEET_STATE.snapshot()
    feats = snap["latest_features"].get(machine_id)
    if not feats:
        return jsonify({"error": "unknown machine_id"}), 404
    result = score_features(feats)
    return jsonify({
        "machine_id": machine_id,
        "result": result,
        "raw_history": snap["raw_history"].get(machine_id, []),
        "risk_history": snap["risk_history"].get(machine_id, []),
        "latest_features": feats,
    })


@app.route("/api/lookup_timestamp", methods=["POST"])
def lookup_timestamp():
    ts = request.json.get("timestamp")
    matches = REPLAY_DF.index[REPLAY_DF["Timestamp"] == ts].tolist()
    if not matches:
        return jsonify({"error": "timestamp not found. Valid range: "
                        f"{REPLAY_DF['Timestamp'].iloc[0]} to {REPLAY_DF['Timestamp'].iloc[-1]}, "
                        "1-minute intervals."}), 404
    idx = matches[0]
    row = REPLAY_DF.iloc[idx]
    feats = FULL_FEATS[idx]
    result = score_features(feats)
    return jsonify({
        "timestamp": ts,
        "actual_failure_label": int(row["Machine failure"]),
        "actual_failure_types": {k: int(row[k]) for k in ["TWF", "HDF", "PWF", "OSF", "RNF"]},
        "result": result,
        "features": feats,
    })


@app.route("/api/manual_predict", methods=["POST"])
def manual_predict():
    """Mode 1: manual raw entry. Optionally borrow history from a timestamp
    so rolling/lag features aren't undefined (cold-start otherwise)."""
    data = request.json
    borrow_ts = data.get("borrow_history_timestamp")
    buf = MachineBuffer("manual_temp")

    if borrow_ts:
        row = REPLAY_DF[REPLAY_DF["Timestamp"] == borrow_ts]
        if not row.empty:
            row = row.iloc[0]
            session_start = REPLAY_DF[(REPLAY_DF["tool_cycle"] == 0) &
                                       (REPLAY_DF.index <= row.name)].index.max()
            session = REPLAY_DF.iloc[session_start:row.name]  # history BEFORE this point
            for _, r in session.iterrows():
                reading = {
                    "air_temp": r["Air temperature [K]"], "process_temp": r["Process temperature [K]"],
                    "rpm": r["Rotational speed [rpm]"], "torque": r["Torque [Nm]"],
                    "tool_wear": r["Tool wear [min]"], "type": int(r["Type"]),
                }
                buf.push(reading)

    reading = {
        "air_temp": float(data["air_temp"]), "process_temp": float(data["process_temp"]),
        "rpm": float(data["rpm"]), "torque": float(data["torque"]),
        "tool_wear": float(data["tool_wear"]), "type": TYPE_MAP.get(data.get("type", "M"), 1),
    }
    feats = buf.push(reading)
    result = score_features(feats)
    return jsonify({
        "result": result,
        "features": feats,
        "used_borrowed_history": bool(borrow_ts),
        "cold_start_warning": (not borrow_ts and buf.history and len(buf.history) == 1),
        "entered_values": {
            "air_temp": reading["air_temp"], "process_temp": reading["process_temp"],
            "rpm": reading["rpm"], "torque": reading["torque"], "tool_wear": reading["tool_wear"],
        },
        "normal_ranges": NORMAL_RANGES,
    })


@app.route("/api/timestamp_range", methods=["POST"])
def timestamp_range():
    """Mode 2 (rebuilt): free-form 'from this timestamp to this timestamp'
    lookup directly against the real dataset. Uses the pre-cached, causally
    correct features/probabilities for the whole historical stream, so this
    is fast and guaranteed consistent with training."""
    data = request.json
    start_ts, end_ts = data.get("start_timestamp"), data.get("end_timestamp")
    idx = REPLAY_DF.index[(REPLAY_DF["Timestamp"] >= start_ts) & (REPLAY_DF["Timestamp"] <= end_ts)]
    if len(idx) == 0:
        return jsonify({"error": "no rows in that range. Valid range: "
                        f"{REPLAY_DF['Timestamp'].iloc[0]} to {REPLAY_DF['Timestamp'].iloc[-1]}"}), 404

    idx = list(idx)
    probs = FULL_PROBA[idx]
    tiers = [tier_for(p) for p in probs]
    tier_counts = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
    for t in tiers:
        tier_counts[t] += 1

    seg = REPLAY_DF.iloc[idx]
    trajectory = [{
        "timestamp": seg["Timestamp"].iloc[i],
        "probability": float(probs[i]),
        "risk_tier": tiers[i],
        "actual_failure": int(seg["Machine failure"].iloc[i]),
    } for i in range(len(idx))]

    # aggregate SHAP over the range (subsample if large, to stay fast)
    sample_idx = idx if len(idx) <= 300 else list(np.random.RandomState(0).choice(idx, 300, replace=False))
    Xs = FULL_X.iloc[sample_idx]
    shap_values = EXPLAINER.shap_values(Xs)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_abs = np.abs(sv).mean(axis=0)
    ranked = sorted(zip(FEATURE_COLUMNS, mean_abs.tolist()), key=lambda kv: kv[1], reverse=True)
    top_drivers_overall = [{"feature": k, "avg_abs_impact": v} for k, v in ranked[:5]]

    # dominant risk-flag counts across the range -> recommendation
    flag_cols = ["twf_risk_flag", "hdf_risk_flag", "pwf_risk_flag", "osf_risk_flag"]
    flag_counts = {c: int(sum(FULL_FEATS[i][c] for i in idx)) for c in flag_cols}
    dominant_flag = max(flag_counts, key=flag_counts.get) if any(flag_counts.values()) else None
    recommendation = (ACTION_MAP[dominant_flag] if dominant_flag and flag_counts[dominant_flag] > 0
                       else "No physics thresholds breached in this window — routine monitoring sufficient.")

    env_fields = ["ambient_temp", "ambient_humidity", "atmospheric_pressure",
                  "grid_voltage_fluctuation", "factory_load_density",
                  "operator_skill_proxy", "particulate_matter_pm10", "ambient_vibration_noise"]
    env_avg = {f: float(seg[f].mean()) for f in env_fields if f in seg.columns}

    return jsonify({
        "start_timestamp": start_ts, "end_timestamp": end_ts, "n_rows": len(idx),
        "avg_probability": float(np.mean(probs)), "max_probability": float(np.max(probs)),
        "tier_counts": tier_counts,
        "actual_failures_in_range": int(seg["Machine failure"].sum()),
        "actual_failure_type_counts": {c: int(seg[c].sum()) for c in ["TWF", "HDF", "PWF", "OSF", "RNF"]},
        "flag_counts_in_range": flag_counts,
        "recommendation": recommendation,
        "top_drivers_overall": top_drivers_overall,
        "environment_avg": env_avg,
        "trajectory": trajectory,
    })


N_SESSION_CHUNKS = 40


def _session_bounds(idx):
    chunk_len = len(REPLAY_DF) // N_SESSION_CHUNKS
    start = idx * chunk_len
    end = start + chunk_len if idx < N_SESSION_CHUNKS - 1 else len(REPLAY_DF)
    return start, end


@app.route("/api/machine_range", methods=["POST"])
def machine_range():
    """Fixed 40 equal time-block 'virtual machine' replay (kept for the Session
    Replay tab). NOTE: tool_cycle in this dataset ramps 0->119 once for the
    whole file rather than resetting per machine, so sessions here are
    equal-length contiguous time blocks of the single real stream, not
    separate physical machines. Documented rather than silently pretended
    otherwise. Uses the same precomputed cache as timestamp_range."""
    data = request.json
    session_idx = int(data.get("session_index", 0))
    if session_idx >= N_SESSION_CHUNKS or session_idx < 0:
        return jsonify({"error": "invalid session_index"}), 400
    start, end = _session_bounds(session_idx)
    seg = REPLAY_DF.iloc[start:end]
    probs = FULL_PROBA[start:end]
    trajectory = [{
        "timestamp": seg["Timestamp"].iloc[i],
        "probability": float(probs[i]),
        "risk_tier": tier_for(probs[i]),
        "actual_failure": int(seg["Machine failure"].iloc[i]),
    } for i in range(len(seg))]
    return jsonify({"session_index": session_idx, "n_rows": len(seg), "trajectory": trajectory})


@app.route("/api/sessions")
def list_sessions():
    out = []
    for i in range(N_SESSION_CHUNKS):
        s, e = _session_bounds(i)
        seg = REPLAY_DF.iloc[s:e]
        out.append({
            "session_index": i,
            "start_timestamp": seg["Timestamp"].iloc[0],
            "end_timestamp": seg["Timestamp"].iloc[-1],
            "n_rows": len(seg),
            "n_failures": int(seg["Machine failure"].sum()),
            "had_failure": int(seg["Machine failure"].sum() > 0),
        })
    return jsonify(out)


@app.route("/api/global_insights")
def global_insights():
    importances = MODEL.feature_importances_
    ranked = sorted(zip(FEATURE_COLUMNS, importances.tolist()), key=lambda kv: kv[1], reverse=True)
    return jsonify({
        "model_metrics": MODEL_METRICS,
        "global_feature_importance": [{"feature": k, "importance": v} for k, v in ranked[:15]],
        "roc_curve": MODEL_INSIGHTS["roc_curve"],
        "pr_curve": MODEL_INSIGHTS["pr_curve"],
        "probability_histogram": MODEL_INSIGHTS["probability_histogram"],
        "failure_type_counts": MODEL_INSIGHTS["failure_type_counts"],
        "env_correlation": MODEL_INSIGHTS["env_correlation"],
    })


@app.route("/api/model_metrics")
def model_metrics():
    return jsonify(MODEL_METRICS)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False, use_reloader=False)
