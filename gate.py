# gate.py
import numpy as np
import pandas as pd


def _col(X: pd.DataFrame, name: str, default=0.0):
    """Safe column getter."""
    if name in X.columns:
        return pd.to_numeric(X[name], errors="coerce").fillna(default).values.astype(float)
    return np.full(len(X), float(default), dtype=float)


def apply_gate(X: pd.DataFrame, drop_key: str = "A"):
    """
    Returns: (X, mask)

    IMPORTANT:
    - mask must be different for A/B/C so A/B/C results can differ.
    - This mask is later used to suppress risk_score (probs *= mask).
    """
    if not isinstance(X, pd.DataFrame) or len(X) == 0:
        return X, None

    drop_key = (drop_key or "B").strip().upper()

    # Core signals from feature matrix (as trained)
    map_filled = _col(X, "MAP_filled", default=np.nan)
    map_drop_2m = _col(X, "MAP_drop_2m", default=0.0)  # negative means drop
    map_d60 = _col(X, "MAP_d60", default=0.0)          # delta over ~60 window
    map_d1 = _col(X, "MAP_d1", default=0.0)            # delta per step (approx)
    hr = _col(X, "HR", default=np.nan)
    spo2 = _col(X, "SpO2", default=np.nan)

    # Basic plausibility mask (optional)
    # Keep it permissive to avoid overblocking
    valid = np.ones(len(X), dtype=bool)
    if np.isfinite(map_filled).any():
        valid &= np.isfinite(map_filled)
    if np.isfinite(hr).any():
        valid &= (hr > 20) & (hr < 220)
    if np.isfinite(spo2).any():
        valid &= (spo2 > 40) & (spo2 <= 100)

    # ----------------------------
    # A: Rapid drop gate
    # ----------------------------
    # Rapid hypotension signature: big negative drop in short time
    gate_A = (map_drop_2m <= -12) | (map_d1 <= -6)

    # ----------------------------
    # B: Gradual drop gate
    # ----------------------------
    # Gradual: not huge sudden drop, but drifting downward (negative trend)
    gate_B = (map_drop_2m > -12) & (map_d60 <= -6)

    # ----------------------------
    # C: Intermittent drop gate
    # ----------------------------
    # Intermittent: oscillatory / repeated dips: higher local variability
    # Use a proxy: absolute step changes + moderate net trend
    gate_C = (np.abs(map_d1) >= 4) & (np.abs(map_d60) < 6) & (map_drop_2m > -18)

    if drop_key == "A":
        gate = gate_A
    elif drop_key == "B":
        gate = gate_B
    else:
        gate = gate_C

    mask = (valid & gate).astype(bool)

    # Return X unchanged + mask
    return X, mask
