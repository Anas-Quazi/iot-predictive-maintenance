"""
feature_engine.py

SINGLE SOURCE OF TRUTH for feature computation.
Used identically by: training, the live simulator, manual "what-if" input,
and (eventually) a real sensor gateway. This guarantees zero train/serve skew.

Design rules (why it looks the way it does):
- Every feature here is CAUSAL: it only uses the current reading + past
  readings in the buffer, never future ones. This is what makes it safe
  to compute identically offline (training) and online (live stream).
- Raw sensor inputs required per reading:
    air_temp (K), process_temp (K), rpm, torque (Nm), tool_wear (min), type (L/M/H)
  Optional context inputs (default to sane values if missing):
    ambient_temp, ambient_humidity, atmospheric_pressure, grid_voltage_fluctuation,
    factory_load_density, operator_skill_proxy, particulate_matter_pm10,
    ambient_vibration_noise
- OSF / HDF / PWF thresholds below are the published AI4I-2020 synthetic
  failure-generation rules (domain physics), NOT derived from the label.
  Using them as engineered inputs is legitimate feature engineering, not leakage.
"""

from collections import deque
import numpy as np

RAW_NUMERIC = ["air_temp", "process_temp", "rpm", "torque", "tool_wear"]
ENV_FIELDS = [
    "ambient_temp", "ambient_humidity", "atmospheric_pressure",
    "grid_voltage_fluctuation", "factory_load_density",
    "operator_skill_proxy", "particulate_matter_pm10", "ambient_vibration_noise",
]
ENV_DEFAULTS = {
    "ambient_temp": 293.0, "ambient_humidity": 55.0, "atmospheric_pressure": 1013.0,
    "grid_voltage_fluctuation": 0.0, "factory_load_density": 0.5,
    "operator_skill_proxy": 3, "particulate_matter_pm10": 40.0,
    "ambient_vibration_noise": 1.0,
}
TYPE_MAP = {"L": 0, "M": 1, "H": 2}
OSF_THRESHOLD = {0: 11000, 1: 12000, 2: 13000}  # AI4I published rule, per product type

WINDOWS = (5, 15)
BUFFER_LEN = 30  # must be >= max(WINDOWS)


def _osf_threshold(type_code):
    return OSF_THRESHOLD.get(int(type_code), 12000)


def instant_features(reading: dict) -> dict:
    """Features computable from a single reading, no history required."""
    air = reading["air_temp"]
    proc = reading["process_temp"]
    rpm = reading["rpm"]
    torque = reading["torque"]
    wear = reading["tool_wear"]
    type_code = TYPE_MAP.get(reading.get("type", "M"), 1) if isinstance(reading.get("type", 1), str) else int(reading.get("type", 1))

    power = torque * rpm
    temp_diff = proc - air
    torque_per_wear = torque / (wear + 1.0)
    strain_index = torque * wear
    power_per_temp = power / (temp_diff if temp_diff != 0 else 0.1)

    twf_flag = int(200 <= wear <= 240)
    hdf_flag = int((temp_diff < 8.6) and (rpm < 1380))
    pwf_flag = int((power < 3500) or (power > 9000))
    osf_flag = int(strain_index > _osf_threshold(type_code))
    risk_score = twf_flag + hdf_flag + pwf_flag + osf_flag

    out = {
        "air_temp": air, "process_temp": proc, "rpm": rpm, "torque": torque,
        "tool_wear": wear, "type_code": type_code,
        "power": power, "temp_diff": temp_diff, "torque_per_wear": torque_per_wear,
        "strain_index": strain_index, "power_per_temp": power_per_temp,
        "twf_risk_flag": twf_flag, "hdf_risk_flag": hdf_flag,
        "pwf_risk_flag": pwf_flag, "osf_risk_flag": osf_flag,
        "risk_score": risk_score,
    }
    for f in ENV_FIELDS:
        out[f] = reading.get(f, ENV_DEFAULTS[f])
    return out


class MachineBuffer:
    """Rolling causal history for ONE machine. Feed raw readings in order."""

    def __init__(self, machine_id):
        self.machine_id = machine_id
        self.history = deque(maxlen=BUFFER_LEN)  # of instant_features dicts

    def push(self, reading: dict) -> dict:
        feats = instant_features(reading)
        self.history.append(feats)
        return self.compute_features()

    def compute_features(self) -> dict:
        """Full feature vector for the MOST RECENT reading in the buffer."""
        if not self.history:
            raise ValueError("No readings yet for this machine")

        hist = list(self.history)
        cur = hist[-1]
        out = dict(cur)

        base_cols = ["air_temp", "process_temp", "rpm", "torque", "power", "temp_diff"]
        for w in WINDOWS:
            window = hist[-w:]
            for col in base_cols:
                vals = np.array([h[col] for h in window], dtype=float)
                out[f"{col}_mean_{w}"] = float(vals.mean())
                out[f"{col}_std_{w}"] = float(vals.std()) if len(vals) > 1 else 0.0

        # lags
        for col in ["torque", "rpm", "power"]:
            out[f"{col}_lag_1"] = hist[-2][col] if len(hist) >= 2 else cur[col]
            out[f"{col}_lag_2"] = hist[-3][col] if len(hist) >= 3 else cur[col]

        # rolling z-scores (adaptive, causal — uses the 15-window stats just computed)
        for col in ["torque", "rpm", "power", "tool_wear", "temp_diff"]:
            mean_key, std_key = f"{col}_mean_15", f"{col}_std_15"
            if mean_key not in out:  # tool_wear/temp_diff not in base_cols loop above w=15 fallback
                vals = np.array([h[col] for h in hist[-15:]], dtype=float)
                m, s = float(vals.mean()), float(vals.std()) if len(vals) > 1 else 0.0
            else:
                m, s = out[mean_key], out[std_key]
            out[f"{col}_zscore"] = (cur[col] - m) / s if s > 1e-6 else 0.0

        out["machine_id"] = self.machine_id
        out["readings_seen"] = len(hist)
        return out


FEATURE_COLUMNS = None  # set by build_feature_columns() after first computation


def build_feature_columns(sample_feature_dict):
    """Deterministic, sorted (minus id/meta) column order used by the model."""
    exclude = {"machine_id", "readings_seen", "type"}
    cols = sorted([k for k in sample_feature_dict.keys() if k not in exclude])
    return cols


def to_vector(feat_dict, columns):
    return [feat_dict.get(c, 0.0) for c in columns]
