import numpy as np
import pandas as pd

# feature list = 26 columns (كما في feature_cols.joblib)
# ['EtCO2_d1','EtCO2_d60','EtCO2_m30','EtCO2_m60','EtCO2_s60',
#  'HR_d1','HR_d60','HR_m30','HR_m60','HR_s60',
#  'MAP_d1','MAP_d60','MAP_drop_2m','MAP_m30','MAP_m60','MAP_s60',
#  'RR_d1','RR_d60','RR_m30','RR_m60','RR_s60',
#  'SpO2_d1','SpO2_d60','SpO2_m30','SpO2_m60','SpO2_s60']

SIGNALS = ["MAP", "HR", "SpO2", "RR", "EtCO2"]

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # required minimal
    if "time" not in df.columns:
        raise ValueError("CSV must contain 'time' column.")
    for c in ["MAP", "HR", "SpO2"]:
        if c not in df.columns:
            raise ValueError(f"CSV must contain '{c}' column.")
    # optional
    if "RR" not in df.columns:
        df["RR"] = np.nan
    if "EtCO2" not in df.columns:
        df["EtCO2"] = np.nan
    return df

def _estimate_dt(df: pd.DataFrame) -> float:
    t = pd.to_numeric(df["time"], errors="coerce").to_numpy()
    t = t[np.isfinite(t)]
    if len(t) < 3:
        return 1.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return 1.0
    med = float(np.median(dt))
    return 1.0 if (not np.isfinite(med) or med <= 0) else med

def _roll_mean(x: pd.Series, win: int) -> pd.Series:
    return x.rolling(win, min_periods=1).mean()

def _roll_std(x: pd.Series, win: int) -> pd.Series:
    return x.rolling(win, min_periods=1).std(ddof=0)

def extract_features_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df columns: time, MAP, HR, SpO2 (RR optional, EtCO2 optional)
    Output: DataFrame with 26 features aligned per-row.
    """
    df = _ensure_columns(df)
    df = df.copy()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)

    dt = _estimate_dt(df)
    # windows in samples
    w30 = max(1, int(round(30.0 / dt)))
    w60 = max(1, int(round(60.0 / dt)))
    w120 = max(1, int(round(120.0 / dt)))

    feats = {}

    for s in SIGNALS:
        x = pd.to_numeric(df[s], errors="coerce")
        m30 = _roll_mean(x, w30)
        m60 = _roll_mean(x, w60)
        s60 = _roll_std(x, w60)

        # deltas: now - 1s and now - 60s (approx by samples)
        d1  = x - x.shift(max(1, int(round(1.0 / dt))))
        d60 = x - x.shift(w60)

        feats[f"{s}_m30"] = m30
        feats[f"{s}_m60"] = m60
        feats[f"{s}_s60"] = s60
        feats[f"{s}_d1"]  = d1
        feats[f"{s}_d60"] = d60

    # MAP_drop_2m: current MAP - MAP 2 minutes ago (negative = drop)
    MAP = pd.to_numeric(df["MAP"], errors="coerce")
    feats["MAP_drop_2m"] = MAP - MAP.shift(w120)

    X = pd.DataFrame(feats)
    # replace inf
    X = X.replace([np.inf, -np.inf], np.nan)
    # fill NaNs (forward then 0)
    X = X.ffill().fillna(0.0)

    return X
