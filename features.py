import numpy as np
import pandas as pd

# Expected columns (must match feature_cols.joblib exactly)
FEATURES = [
    "EtCO2_d1", "EtCO2_d60", "EtCO2_m30", "EtCO2_m60", "EtCO2_s60",
    "HR_d1", "HR_d60", "HR_m30", "HR_m60", "HR_s60",
    "MAP_d1", "MAP_d60", "MAP_drop_2m", "MAP_m30", "MAP_m60", "MAP_s60",
    "RR_d1", "RR_d60", "RR_m30", "RR_m60", "RR_s60",
    "SpO2_d1", "SpO2_d60", "SpO2_m30", "SpO2_m60", "SpO2_s60",
]

VITALS = ["MAP", "HR", "SpO2", "RR", "EtCO2"]

def _safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df must include: time, MAP, HR, SpO2 (RR optional) (EtCO2 optional)
    Returns a DataFrame with FEATURE columns exactly in order.
    """
    df = df.copy()

    # Ensure required columns exist
    if "time" not in df.columns:
        df["time"] = np.arange(len(df), dtype=float)

    for v in VITALS:
        if v not in df.columns:
            df[v] = np.nan
        df[v] = _safe_float_series(df[v])

    # estimate dt in seconds from time column (fallback to 1.0)
    t = _safe_float_series(df["time"])
    if len(t) >= 3:
        dt = float(np.nanmedian(np.diff(t)))
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0
    else:
        dt = 1.0

    def win(seconds: float) -> int:
        w = int(round(seconds / dt))
        return max(w, 1)

    w30 = win(30)
    w60 = win(60)
    w120 = win(120)

    feats = pd.DataFrame(index=df.index)

    # Build rolling stats + deltas
    for v in VITALS:
        s = df[v]
        feats[f"{v}_m30"] = s.rolling(w30, min_periods=1).mean()
        feats[f"{v}_m60"] = s.rolling(w60, min_periods=1).mean()
        feats[f"{v}_s60"] = s.rolling(w60, min_periods=2).std()
        feats[f"{v}_d1"] = s.diff(1)
        feats[f"{v}_d60"] = s.diff(w60)

    # Special MAP feature: drop over last 2 minutes
    map_roll_max_2m = df["MAP"].rolling(w120, min_periods=1).max()
    feats["MAP_drop_2m"] = (map_roll_max_2m - df["MAP"]).clip(lower=0)

    # Keep only expected features and order them
    for c in FEATURES:
        if c not in feats.columns:
            feats[c] = np.nan

    feats = feats[FEATURES].copy()

    return feats
