import numpy as np
import pandas as pd

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal feature extractor (safe fallback).
    Assumes df has at least: MAP, HR (optionally SpO2, EtCO2, RR).
    Returns a feature DataFrame per-row (same length as df).
    """
    out = pd.DataFrame(index=df.index)

    # Basic signals
    for col in ["MAP", "HR", "SpO2", "EtCO2", "RR"]:
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce")
            out[f"{col}_val"] = x
            out[f"{col}_m30"] = x.rolling(30, min_periods=1).mean()
            out[f"{col}_m60"] = x.rolling(60, min_periods=1).mean()
            out[f"{col}_s60"] = x.rolling(60, min_periods=2).std().fillna(0)

            out[f"{col}_d1"] = x.diff(1).fillna(0)
            out[f"{col}_d60"] = x.diff(60).fillna(0)
        else:
            # If missing, add safe zeros
            out[f"{col}_val"] = 0.0
            out[f"{col}_m30"] = 0.0
            out[f"{col}_m60"] = 0.0
            out[f"{col}_s60"] = 0.0
            out[f"{col}_d1"] = 0.0
            out[f"{col}_d60"] = 0.0

    # A simple MAP drop feature (like your pipeline idea)
    mapx = pd.to_numeric(df.get("MAP", 0), errors="coerce").fillna(method="ffill").fillna(0)
    out["MAP_drop_2m"] = (mapx - mapx.shift(120)).fillna(0)  # if 1Hz, ~2 minutes

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
    return out
