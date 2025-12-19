import pandas as pd
import numpy as np

def _safe_series(df, col):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([np.nan] * len(df))

def extract_features_from_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal feature extractor for dashboard inference.
    IMPORTANT: we will align to feature_cols.joblib later, so missing features become 0.
    """
    d = df.copy()

    map_s = _safe_series(d, "MAP")
    hr_s = _safe_series(d, "HR")
    spo2_s = _safe_series(d, "SpO2")
    rr_s = _safe_series(d, "RR")

    # rolling stats (simple + stable)
    def roll_mean(x, w=30): return x.rolling(w, min_periods=1).mean()
    def roll_std(x, w=30):  return x.rolling(w, min_periods=2).std()

    feats = pd.DataFrame({
        "MAP": map_s,
        "HR": hr_s,
        "SpO2": spo2_s,
        "RR": rr_s,

        "MAP_m30": roll_mean(map_s, 30),
        "MAP_s30": roll_std(map_s, 30),
        "HR_m30": roll_mean(hr_s, 30),
        "HR_s30": roll_std(hr_s, 30),
        "SpO2_m30": roll_mean(spo2_s, 30),
        "SpO2_s30": roll_std(spo2_s, 30),

        # simple delta features
        "MAP_d1": map_s.diff(1),
        "MAP_d10": map_s.diff(10),
        "HR_d1": hr_s.diff(1),
        "SpO2_d1": spo2_s.diff(1),

        # drop over recent window
        "MAP_drop_2m": (roll_mean(map_s, 1) - roll_mean(map_s, 120)),
    })

    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feats

def extract_features_from_single_row(row: dict) -> pd.DataFrame:
    """
    Single-row input → 1-row feature DataFrame.
    """
    m = float(row.get("MAP", 0))
    h = float(row.get("HR", 0))
    s = float(row.get("SpO2", 0))
    r = float(row.get("RR", 0))

    feats = pd.DataFrame([{
        "MAP": m, "HR": h, "SpO2": s, "RR": r,
        "MAP_m30": m, "MAP_s30": 0.0,
        "HR_m30": h,  "HR_s30": 0.0,
        "SpO2_m30": s, "SpO2_s30": 0.0,
        "MAP_d1": 0.0, "MAP_d10": 0.0,
        "HR_d1": 0.0, "SpO2_d1": 0.0,
        "MAP_drop_2m": 0.0,
    }])
    return feats
