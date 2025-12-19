import numpy as np
import pandas as pd

def _estimate_dt_seconds(time_arr: np.ndarray) -> float:
    t = pd.to_numeric(pd.Series(time_arr), errors="coerce").dropna().values
    if len(t) < 3:
        return 1.0
    diffs = np.diff(t)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return 1.0
    return float(np.median(diffs))

def _rolling_by_seconds(x: pd.Series, dt: float, sec: float, func="mean"):
    win = max(1, int(round(sec / max(dt, 1e-6))))
    r = x.rolling(win, min_periods=1)
    if func == "mean":
        return r.mean()
    if func == "std":
        return r.std(ddof=0).fillna(0)
    if func == "min":
        return r.min()
    if func == "max":
        return r.max()
    return r.mean()

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    # Expect columns: time, MAP, HR, SpO2, RR (RR optional)
    d = df.copy()

    # sort by time
    d = d.sort_values("time").reset_index(drop=True)

    dt = _estimate_dt_seconds(d["time"].values)
    d["dt_used"] = dt

    # Ensure numeric
    for col in ["MAP", "HR", "SpO2", "RR"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    # Fill forward for stability (then 0)
    for col in ["MAP", "HR", "SpO2", "RR"]:
        d[col] = d[col].ffill().fillna(0)

    feats = pd.DataFrame(index=d.index)

    # Basic stats (like your earlier pipeline style)
    for v in ["MAP", "HR", "SpO2", "RR"]:
        x = d[v].astype(float)

        feats[f"{v}_m30"] = _rolling_by_seconds(x, dt, 30, "mean")
        feats[f"{v}_m60"] = _rolling_by_seconds(x, dt, 60, "mean")
        feats[f"{v}_s60"] = _rolling_by_seconds(x, dt, 60, "std")

        feats[f"{v}_d1"] = x.diff(1).fillna(0)
        shift60 = max(1, int(round(60 / max(dt, 1e-6))))
        feats[f"{v}_d60"] = (x - x.shift(shift60)).fillna(0)

    # MAP drop over 2 minutes (useful gate signal)
    shift2m = max(1, int(round(120 / max(dt, 1e-6))))
    feats["MAP_drop_2m"] = (d["MAP"] - d["MAP"].shift(shift2m)).fillna(0)

    # Keep time index (optional)
    feats["time"] = d["time"].values

    return feats
