# features.py
import numpy as np
import pandas as pd


def get_expected_feature_columns(model):
    """
    Use the model's own expected feature names (most reliable).
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    # fallback
    return ["MAP_filled", "HR", "SpO2", "RR", "EtCO2"]


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def _infer_dt(time_arr: np.ndarray) -> float:
    """Infer median sampling interval from time column."""
    if time_arr is None or len(time_arr) < 3:
        return 1.0
    diffs = np.diff(time_arr.astype(float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return 1.0
    dt = float(np.median(diffs))
    return 1.0 if (not np.isfinite(dt) or dt <= 0) else dt


def _steps_for_seconds(seconds: float, dt: float) -> int:
    return max(1, int(round(float(seconds) / float(dt))))


def _shift(arr: np.ndarray, k: int):
    """Shift forward by k (current - past)."""
    s = pd.Series(arr)
    return s.shift(k).to_numpy()


def _diff(arr: np.ndarray, k: int):
    """arr - arr shifted by k."""
    return arr - _shift(arr, k)


def _slope(arr: np.ndarray, win: int):
    """
    Simple slope proxy over window:
    slope ≈ (x[t] - x[t-win]) / win
    """
    if win <= 0:
        win = 1
    d = _diff(arr, win)
    return d / float(win)


def build_feature_matrix(df_raw: pd.DataFrame, expected_cols: list) -> pd.DataFrame:
    """
    Build X that matches EXACTLY the columns the model was trained on.

    It computes common engineered features used in your model:
    - *_m30, *_m60  : lag (previous value)
    - *_s60         : slope over 60 seconds window (proxy)
    - *_d1          : step difference (current - prev)
    - *_d60         : difference over 60 seconds
    - MAP_drop_2m   : difference over ~2 minutes (current - value 2 minutes ago)
    """
    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]

    # Required columns
    if "time" not in df.columns:
        raise ValueError("CSV must contain a 'time' column.")
    for base in ["MAP", "HR", "SpO2"]:
        if base not in df.columns:
            raise ValueError("CSV must contain MAP, HR, SpO2 (and time).")

    # Optional
    if "RR" not in df.columns:
        df["RR"] = np.nan
    if "EtCO2" not in df.columns:
        df["EtCO2"] = np.nan

    # numeric + sort
    df["time"] = _to_num(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # base signals
    df["MAP_filled"] = _to_num(df["MAP"]).ffill().bfill()
    df["HR"] = _to_num(df["HR"])
    df["SpO2"] = _to_num(df["SpO2"])
    df["RR"] = _to_num(df["RR"])
    df["EtCO2"] = _to_num(df["EtCO2"])

    time_arr = df["time"].to_numpy(dtype=float)
    dt = _infer_dt(time_arr)

    # window steps
    k30 = _steps_for_seconds(30.0, dt)
    k60 = _steps_for_seconds(60.0, dt)
    k120 = _steps_for_seconds(120.0, dt)  # 2 minutes

    # Helper to generate features for any signal name
    def add_features(sig_name: str):
        arr = df[sig_name].to_numpy(dtype=float)

        # lags (previous value)
        df[f"{sig_name}_m30"] = _shift(arr, k30)
        df[f"{sig_name}_m60"] = _shift(arr, k60)

        # slope proxy over 60s
        df[f"{sig_name}_s60"] = _slope(arr, k60)

        # diffs
        df[f"{sig_name}_d1"] = _diff(arr, 1)
        df[f"{sig_name}_d60"] = _diff(arr, k60)

    # Generate for each vital (as in expected list)
    # Note: model expects MAP_*, HR_*, SpO2_*, RR_*, EtCO2_*
    # MAP is stored under MAP_filled for base, but engineered names are MAP_* (your expected list uses MAP_*).
    # لذلك ننشئ MAP features من MAP_filled ولكن بأسماء MAP_*
    map_arr = df["MAP_filled"].to_numpy(dtype=float)
    df["MAP_m30"] = _shift(map_arr, k30)
    df["MAP_m60"] = _shift(map_arr, k60)
    df["MAP_s60"] = _slope(map_arr, k60)
    df["MAP_d1"] = _diff(map_arr, 1)
    df["MAP_d60"] = _diff(map_arr, k60)
    df["MAP_drop_2m"] = _diff(map_arr, k120)  # current - 2min ago

    # HR / SpO2 / RR / EtCO2 features
    add_features("HR")
    add_features("SpO2")
    add_features("RR")
    add_features("EtCO2")

    # Final X in exact trained order
    X = pd.DataFrame(index=df.index)
    for c in expected_cols:
        if c in df.columns:
            X[c] = df[c]
        else:
            # If model expects something else, create it as NaN (safer than 0)
            # pipeline imputer will handle it
            X[c] = np.nan

    return X
