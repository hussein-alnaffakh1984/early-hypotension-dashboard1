# features.py
import numpy as np
import pandas as pd

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df must contain: time, MAP, HR, SpO2 (RR optional).
    Returns features DataFrame aligned per-row (timeseries features).
    """
    d = df.copy()

    # ensure numeric
    for c in ["MAP", "HR", "SpO2", "RR"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # basic rolling windows (in samples, not seconds)
    # Works for both 1Hz and irregular data as a simple baseline
    def roll_mean(x, w): return x.rolling(w, min_periods=1).mean()
    def roll_std(x, w):  return x.rolling(w, min_periods=1).std().fillna(0)

    feat = pd.DataFrame(index=d.index)

    # Core signals
    feat["MAP"]  = d["MAP"]
    feat["HR"]   = d["HR"]
    feat["SpO2"] = d["SpO2"]
    feat["RR"]   = d["RR"] if "RR" in d.columns else 0.0

    # Rolling stats
    feat["MAP_m5"]  = roll_mean(d["MAP"], 5)
    feat["MAP_m15"] = roll_mean(d["MAP"], 15)
    feat["MAP_s15"] = roll_std(d["MAP"], 15)

    feat["HR_m5"]   = roll_mean(d["HR"], 5)
    feat["HR_m15"]  = roll_mean(d["HR"], 15)
    feat["HR_s15"]  = roll_std(d["HR"], 15)

    feat["SpO2_m5"]  = roll_mean(d["SpO2"], 5)
    feat["SpO2_m15"] = roll_mean(d["SpO2"], 15)
    feat["SpO2_s15"] = roll_std(d["SpO2"], 15)

    # Simple deltas (trend)
    feat["MAP_d1"] = d["MAP"].diff().fillna(0)
    feat["HR_d1"]  = d["HR"].diff().fillna(0)
    feat["SpO2_d1"]= d["SpO2"].diff().fillna(0)

    # Fill remaining NaNs
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(0)

    return feat
