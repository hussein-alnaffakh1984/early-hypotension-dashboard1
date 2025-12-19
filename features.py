import numpy as np
import pandas as pd
import joblib

FEATURE_COLS_PATH = "feature_cols.joblib"

def _median_dt_seconds(t: pd.Series) -> float:
    t = pd.to_numeric(t, errors="coerce")
    dt = t.diff().dropna()
    if len(dt) == 0:
        return 1.0
    m = float(dt.median())
    if not np.isfinite(m) or m <= 0:
        return 1.0
    return m

def _ensure_numeric(df: pd.DataFrame, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _rolling_mean(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(window=w, min_periods=1).mean()

def _rolling_std(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(window=w, min_periods=2).std()

def extract_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    ينتج DataFrame Features بأعمدة مطابقة 100% للـ feature_cols.joblib
    (حتى لو RR أو EtCO2 غير موجودين في CSV -> نملأها NaN)
    """
    feature_cols = joblib.load(FEATURE_COLS_PATH)

    df = df_raw.copy()
    if "time" not in df.columns:
        # إذا ماكو time نسوي time تسلسلي
        df["time"] = np.arange(len(df), dtype=float)

    df = _ensure_numeric(df, ["time", "MAP", "HR", "SpO2", "RR", "EtCO2"])
    df = df.sort_values("time").reset_index(drop=True)

    dt = _median_dt_seconds(df["time"])
    w30 = max(1, int(round(30.0 / dt)))
    w60 = max(1, int(round(60.0 / dt)))
    w120 = max(1, int(round(120.0 / dt)))  # 2 minutes

    # إذا العمود ناقص نخلّيه NaN
    for base in ["MAP", "HR", "SpO2", "RR", "EtCO2"]:
        if base not in df.columns:
            df[base] = np.nan

    feats = pd.DataFrame(index=df.index)

    # Helper to build feature set for a signal
    def build_for(sig: str):
        s = df[sig].astype(float)

        feats[f"{sig}_d1"]  = s.diff(1)
        feats[f"{sig}_d60"] = s.diff(w60)

        feats[f"{sig}_m30"] = _rolling_mean(s, w30)
        feats[f"{sig}_m60"] = _rolling_mean(s, w60)

        feats[f"{sig}_s60"] = _rolling_std(s, w60)

    for sig in ["MAP", "HR", "SpO2", "RR", "EtCO2"]:
        build_for(sig)

    # MAP_drop_2m = current MAP - mean(MAP over last 2 minutes)
    map_mean_2m = _rolling_mean(df["MAP"].astype(float), w120)
    feats["MAP_drop_2m"] = df["MAP"].astype(float) - map_mean_2m

    # الآن نضمن نفس الأعمدة وبنفس الترتيب
    for c in feature_cols:
        if c not in feats.columns:
            feats[c] = np.nan

    feats = feats[feature_cols]

    return feats
