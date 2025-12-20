import numpy as np
import pandas as pd
import joblib


def get_expected_feature_columns(model=None, feature_cols_path="feature_cols.joblib"):
    """
    Priority:
      1) feature_cols.joblib (if exists)
      2) model.feature_names_in_ (if available)
    """
    try:
        cols = joblib.load(feature_cols_path)
        return list(cols)
    except Exception:
        pass

    if model is not None and hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    # fallback minimal
    return [
        "MAP_filled", "HR", "SpO2", "EtCO2", "RR"
    ]


def _rolling_mean(s: pd.Series, win: int):
    return s.rolling(window=win, min_periods=1).mean()


def _rolling_std(s: pd.Series, win: int):
    return s.rolling(window=win, min_periods=1).std().fillna(0.0)


def _diff(s: pd.Series, k: int):
    return s.diff(periods=k).fillna(0.0)


def _infer_step_for_minutes(df: pd.DataFrame, minutes: float):
    """
    Infer how many rows correspond to X minutes based on median dt from 'time'.
    If time is in seconds, minutes=2 => 120 seconds.
    If time is in minutes, minutes=2 => 2 minutes.
    """
    t = df["time"].to_numpy()
    if len(t) < 2:
        return 1
    dt = np.median(np.diff(t))
    if dt <= 0 or np.isnan(dt):
        return 1

    # assume minutes unit if dt is ~1 or ~0.5 etc, otherwise might be seconds.
    # We'll compute steps using minutes directly as given.
    steps = int(round(minutes / dt))
    return max(1, steps)


def build_feature_matrix(df_raw: pd.DataFrame, expected_cols: list):
    """
    Build a feature matrix with EXACT expected columns.
    df_raw must already contain: time, MAP, HR, SpO2, and optionally RR, EtCO2.
    """
    df = df_raw.copy()

    # Ensure optional columns exist
    if "RR" not in df.columns:
        df["RR"] = np.nan
    if "EtCO2" not in df.columns:
        df["EtCO2"] = np.nan

    # Fill MAP for stability
    df["MAP_filled"] = df["MAP"].ffill().bfill()

    # infer steps for 30/60 "units" (we treat them as rows if series is short)
    # Many people name m30/m60 as "moving average windows".
    # We implement them as rolling mean window size = min(30, len).
    win30 = min(30, len(df))
    win60 = min(60, len(df))

    # Also derive steps for "drop_2m" and "d60"
    k2m = _infer_step_for_minutes(df, minutes=2.0)
    k60 = min(60, len(df) - 1) if len(df) > 1 else 1

    X = pd.DataFrame(index=df.index)

    # MAP
    X["MAP_filled"] = df["MAP_filled"]
    X["MAP_m30"] = _rolling_mean(df["MAP_filled"], win30)
    X["MAP_m60"] = _rolling_mean(df["MAP_filled"], win60)
    X["MAP_s60"] = _rolling_std(df["MAP_filled"], win60)
    X["MAP_d1"] = _diff(df["MAP_filled"], 1)
    X["MAP_d60"] = _diff(df["MAP_filled"], k60)
    X["MAP_drop_2m"] = df["MAP_filled"] - df["MAP_filled"].shift(k2m)
    X["MAP_drop_2m"] = X["MAP_drop_2m"].fillna(0.0)

    # HR
    X["HR"] = df["HR"]
    X["HR_m30"] = _rolling_mean(df["HR"], win30)
    X["HR_m60"] = _rolling_mean(df["HR"], win60)
    X["HR_s60"] = _rolling_std(df["HR"], win60)
    X["HR_d1"] = _diff(df["HR"], 1)
    X["HR_d60"] = _diff(df["HR"], k60)

    # SpO2
    X["SpO2"] = df["SpO2"]
    X["SpO2_m30"] = _rolling_mean(df["SpO2"], win30)
    X["SpO2_m60"] = _rolling_mean(df["SpO2"], win60)
    X["SpO2_s60"] = _rolling_std(df["SpO2"], win60)
    X["SpO2_d1"] = _diff(df["SpO2"], 1)
    X["SpO2_d60"] = _diff(df["SpO2"], k60)

    # EtCO2 (may be missing -> NaN; keep numeric)
    X["EtCO2"] = df["EtCO2"].astype(float)
    X["EtCO2_m30"] = _rolling_mean(X["EtCO2"], win30)
    X["EtCO2_m60"] = _rolling_mean(X["EtCO2"], win60)
    X["EtCO2_s60"] = _rolling_std(X["EtCO2"], win60)
    X["EtCO2_d1"] = _diff(X["EtCO2"], 1)
    X["EtCO2_d60"] = _diff(X["EtCO2"], k60)

    # RR
    X["RR"] = df["RR"].astype(float)
    X["RR_m30"] = _rolling_mean(X["RR"], win30)
    X["RR_m60"] = _rolling_mean(X["RR"], win60)
    X["RR_s60"] = _rolling_std(X["RR"], win60)
    X["RR_d1"] = _diff(X["RR"], 1)
    X["RR_d60"] = _diff(X["RR"], k60)

    # Reorder and ensure all expected columns exist
    for c in expected_cols:
        if c not in X.columns:
            X[c] = 0.0

    X = X[expected_cols].copy()

    return X
