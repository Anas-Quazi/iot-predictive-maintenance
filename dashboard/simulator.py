"""
simulator.py
Produces RAW sensor readings only (never touches engineered features —
that is feature_engine's job, keeping this swappable for real sensors later).

Two virtual-machine population strategies, blended:
1. Replay: slice the historical dataset by tool_cycle sessions -> each
   session becomes a "virtual machine" ticking forward in time.
2. Drift injection: on top of replayed values, apply a slow upward wear/
   torque/temp trend so degradation is visible and SHAP/probability curves
   actually mean something (not just replaying flat history).

This runs as a background thread and writes into a shared, thread-safe
FleetState object that app.py reads from.
"""
import threading
import time
import random
import pandas as pd
import numpy as np

from feature_engine import MachineBuffer

N_VIRTUAL_MACHINES = 12
TICK_SECONDS = 3


class FleetState:
    def __init__(self):
        self.lock = threading.Lock()
        self.buffers = {}          # machine_id -> MachineBuffer
        self.latest_features = {}  # machine_id -> feature dict
        self.raw_history = {}      # machine_id -> list of raw readings (for trend charts)
        self.risk_history = {}     # machine_id -> list of (timestamp, probability)
        self.alerts = []           # list of dicts, newest first

    def ensure_machine(self, machine_id):
        with self.lock:
            if machine_id not in self.buffers:
                self.buffers[machine_id] = MachineBuffer(machine_id)
                self.raw_history[machine_id] = []
                self.risk_history[machine_id] = []

    def push_reading(self, machine_id, reading):
        self.ensure_machine(machine_id)
        with self.lock:
            feats = self.buffers[machine_id].push(reading)
            self.latest_features[machine_id] = feats
            self.raw_history[machine_id].append(reading)
            if len(self.raw_history[machine_id]) > 60:
                self.raw_history[machine_id] = self.raw_history[machine_id][-60:]
        return feats

    def record_risk(self, machine_id, timestamp, probability):
        with self.lock:
            self.risk_history[machine_id].append({"t": timestamp, "p": probability})
            if len(self.risk_history[machine_id]) > 60:
                self.risk_history[machine_id] = self.risk_history[machine_id][-60:]

    def add_alert(self, machine_id, tier, probability, timestamp):
        with self.lock:
            self.alerts.insert(0, {
                "machine_id": machine_id, "tier": tier,
                "probability": probability, "timestamp": timestamp,
            })
            self.alerts = self.alerts[:50]

    def snapshot(self):
        with self.lock:
            return {
                "latest_features": dict(self.latest_features),
                "raw_history": {k: list(v) for k, v in self.raw_history.items()},
                "risk_history": {k: list(v) for k, v in self.risk_history.items()},
                "alerts": list(self.alerts),
            }


RAW_COL_MAP = {
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
}
ENV_COL_MAP = {
    "ambient_temp": "ambient_temp", "ambient_humidity": "ambient_humidity",
    "atmospheric_pressure": "atmospheric_pressure",
    "grid_voltage_fluctuation": "grid_voltage_fluctuation",
    "factory_load_density": "factory_load_density",
    "operator_skill_proxy": "operator_skill_proxy",
    "particulate_matter_pm10": "particulate_matter_pm10",
    "ambient_vibration_noise": "ambient_vibration_noise",
}


class Simulator(threading.Thread):
    def __init__(self, fleet_state: FleetState, dataset_path="model/replay_source.csv",
                 scorer_callback=None):
        super().__init__(daemon=True)
        self.fleet_state = fleet_state
        self.scorer_callback = scorer_callback  # called after every push: (machine_id, feats)
        self.df = pd.read_csv(dataset_path).sort_values("Timestamp").reset_index(drop=True)
        self._build_sessions()
        self.machine_cursors = {}   # machine_id -> row index within its assigned session
        self.machine_sessions = {}  # machine_id -> session (DataFrame slice)
        self.drift = {}             # machine_id -> cumulative injected drift
        self._assign_sessions()
        self.running = True

    def _build_sessions(self):
        # NOTE: tool_cycle in this dataset does NOT reset repeatedly — it ramps
        # 0->119 once for the whole file. So "sessions" here are equal-length
        # contiguous time blocks of the stream, each acting as one virtual
        # machine's operating window. This is an honest, simple partition of
        # the single real stream into a fleet, not a claim that these are
        # literally different physical machines.
        n_chunks = 40
        chunk_len = len(self.df) // n_chunks
        self.sessions = [self.df.iloc[i * chunk_len:(i + 1) * chunk_len]
                          for i in range(n_chunks)]

    def _assign_sessions(self):
        chosen = random.sample(self.sessions, min(N_VIRTUAL_MACHINES, len(self.sessions)))
        for i, session in enumerate(chosen):
            machine_id = f"M-{i+1:03d}"
            self.machine_sessions[machine_id] = session.reset_index(drop=True)
            self.machine_cursors[machine_id] = 0
            self.drift[machine_id] = 0.0
            # per-machine degradation state — this is what keeps criticals RARE
            # and staggered instead of the whole fleet rising together.
            self.degrading = getattr(self, "degrading", {})
            self.degrade_ticks_left = getattr(self, "degrade_ticks_left", {})
            self.degrade_target = getattr(self, "degrade_target", {})
            self.degrading[machine_id] = False
            self.degrade_ticks_left[machine_id] = 0
            self.degrade_target[machine_id] = 0.0
            self.fleet_state.ensure_machine(machine_id)

    def _update_drift(self, machine_id):
        """Per-tick drift update. Baseline: near-zero drift (normal operation).
        Rarely (small per-tick probability, independent per machine), enter a
        degradation episode of random severity/length, then reset via
        'maintenance'. This keeps the fleet realistically mostly-healthy with
        occasional, staggered elevated machines — matching the ~3.4% real
        failure prevalence — instead of every machine climbing together."""
        if not self.degrading[machine_id]:
            # small chance per tick to start degrading (tuned so, with ~12
            # machines ticking every 3s, elevated events are occasional and
            # staggered, not simultaneous)
            if random.random() < 0.008:
                self.degrading[machine_id] = True
                self.degrade_ticks_left[machine_id] = random.randint(15, 45)
                self.degrade_target[machine_id] = random.uniform(3.0, 14.0)
            else:
                # gentle mean-reverting noise around 0 when healthy
                self.drift[machine_id] = max(0.0, self.drift[machine_id] * 0.85 +
                                              random.uniform(-0.05, 0.05))
        else:
            ticks_left = self.degrade_ticks_left[machine_id]
            target = self.degrade_target[machine_id]
            step = target / max(1, (ticks_left if ticks_left > 0 else 1))
            self.drift[machine_id] = min(target, self.drift[machine_id] + step * random.uniform(0.6, 1.3))
            self.degrade_ticks_left[machine_id] -= 1
            if self.degrade_ticks_left[machine_id] <= 0:
                # "maintenance" — machine gets serviced, drift clears
                self.degrading[machine_id] = False
                self.drift[machine_id] = 0.0
                self.fleet_state.add_alert(
                    machine_id, "MAINTENANCE_RESET", 0.0, pd.Timestamp.now().isoformat())

    def _next_reading(self, machine_id):
        session = self.machine_sessions[machine_id]
        cursor = self.machine_cursors[machine_id]
        if cursor >= len(session):
            cursor = 0
        row = session.iloc[cursor]
        self.machine_cursors[machine_id] = cursor + 1

        self._update_drift(machine_id)
        d = self.drift[machine_id]

        reading = {v: float(row[k]) for k, v in RAW_COL_MAP.items()}
        reading["type"] = int(row["Type"])
        reading["tool_wear"] = max(0.0, reading["tool_wear"] + d * 1.4)
        reading["torque"] = reading["torque"] + d * 0.12
        reading["process_temp"] = reading["process_temp"] + d * 0.015
        for k, v in ENV_COL_MAP.items():
            reading[v] = float(row[k]) if k in row and not pd.isna(row[k]) else None
        return reading

    def run(self):
        while self.running:
            for machine_id in list(self.machine_sessions.keys()):
                reading = self._next_reading(machine_id)
                feats = self.fleet_state.push_reading(machine_id, reading)
                if self.scorer_callback:
                    self.scorer_callback(machine_id, feats)
            time.sleep(TICK_SECONDS)

    def stop(self):
        self.running = False
