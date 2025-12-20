# features.py
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def get_expected_feature_columns(model):
    """
    Use model's expected feature names (best if trained with DataFrame).
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    # fallback
    return [
        "MAP_filled",
        "MAP_m30", "MAP_m60", "MAP_s60", "MAP_d1", "MAP_d60",
        "MAP_drop_2m",
        "HR", "HR_m30", "HR_m60", "HR_s60", "HR_d1", "HR_d60",
        "SpO2", "SpO2_m30", "SpO2_m60", "SpO2_s60", "SpO2_d1", "SpO2_d60",
        "EtCO2", "EtCO2_m30", "EtCO2_m60", "EtCO2_s60", "EtCO2_d1", "EtCO2_d60",
        "RR", "RR_m30", "RR_m60", "RR_s60", "RR_d1", "RR_d60",
    ]


def _to_float_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force time column to float64 ALWAYS (to avoid merge_asof dtype mismatch).
    """
    df = df.copy()
    df["time"] = pd.to_numeric(df["time"], errors="coerce").astype(np.float64)
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


def _ensure_optional_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "RR" not in df.columns:
        df["RR"] = np.nan
    if "EtCO2" not in df.columns:
        df["EtCO2"] = np.nan
    return df


def _resample_to_1s(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Resample irregular time series to 1-second grid using merge_asof.
    ✅ IMPORTANT: time will be float64 on both sides.
    """
    df = df_raw.copy()
    if "time" not in df.columns:
        raise ValueError("Missing required column: time")

    df = _to_float_time(df)
    df = _ensure_optional_signals(df)

    # Convert to "seconds grid" but KEEP as float for safe merge_asof
    t_min = int(np.floor(df["time"].min()))
    t_max = int(np.ceil(df["time"].max()))
    grid = pd.DataFrame({"time": np.arange(t_min, t_max + 1, 1, dtype=np.int64).astype(np.float64)})

    # Remove duplicates by time (keep last)
    df = df.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)

    df2 = pd.merge_asof(
        grid.sort_values("time"),
        df.sort_values("time"),
        on="time",
        direction="nearest"
    )
    return df2


def _roll_feats(s: pd.Series, win: int):
    """
    Rolling mean/std and first difference (slope proxy).
    """
    mean = s.rolling(win, min_periods=max(3, win // 3)).mean()
    std = s.rolling(win, min_periods=max(3, win // 3)).std()
    d1 = s.diff()
    dwin = s.diff(win) / max(1, win)
    return mean, std, d1, dwin


# ---------------------------------------------------------
# Feature matrix for model
# ---------------------------------------------------------
def build_feature_matrix(df_raw: pd.DataFrame, expected_cols: list) -> pd.DataFrame:
    """
    Build X that matches EXACTLY columns used in training.
    Required: time, MAP, HR, SpO2
    Optional: RR, EtCO2
    Produces rolling features with same naming you showed earlier.
    """
    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]

    # Required columns
    required = ["time", "MAP", "HR", "SpO2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV must contain: {missing}")

    df = _to_float_time(df)
    df = _ensure_optional_signals(df)

    # Numeric
    for c in ["MAP", "HR", "SpO2", "RR", "EtCO2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # MAP_filled like training
    df["MAP_filled"] = df["MAP"].ffill().bfill()

    # Use 1-second resampling for stable rolling (optional but recommended)
    df1 = _resample_to_1s(df[["time", "MAP_filled", "HR", "SpO2", "RR", "EtCO2"]])

    # Rolling windows (assuming 1s sampling)
    # m30 = 30s mean, m60=60s mean, s60=60s std, d1=1s diff, d60=60s diff
    def add_group(prefix: str, col: str):
        s = df1[col]
        m30, s30, d1, d30 = _roll_feats(s, 30)
        m60, s60, d1b, d60 = _roll_feats(s, 60)
        # Create names consistent with your expected_cols list
        df1[f"{prefix}_m30"] = m30
        df1[f"{prefix}_m60"] = m60
        df1[f"{prefix}_s60"] = s60
        df1[f"{prefix}_d1"] = d1
        df1[f"{prefix}_d60"] = s.diff(60)

    add_group("MAP", "MAP_filled")
    add_group("HR", "HR")
    add_group("SpO2", "SpO2")
    add_group("RR", "RR")
    add_group("EtCO2", "EtCO2")

    # Drop feature over 2 minutes (120s)
    df1["MAP_drop_2m"] = df1["MAP_filled"].shift(120) - df1["MAP_filled"]

    # Now build X with EXACT expected columns/order
    X = df1.reindex(columns=list(expected_cols), fill_value=np.nan)

    # Ensure numeric float for sklearn pipeline
    X = X.apply(pd.to_numeric, errors="coerce").astype(np.float64)

    return X


# ---------------------------------------------------------
# Drop-type scoring (A/B/C) for AUTO mode
# ---------------------------------------------------------
def compute_drop_scores(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame with:
      time (float64),
      score_A, score_B, score_C
    Logic (simple, explainable):
      A Rapid: big negative slope short-term
      B Gradual: sustained negative slope longer-term
      C Intermittent: high variability + repeated drops (std + abs diff)
    """
    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]
    if "time" not in df.columns or "MAP" not in df.columns:
        raise ValueError("compute_drop_scores needs: time, MAP")

    df = _to_float_time(df)
    df["MAP"] = pd.to_numeric(df["MAP"], errors="coerce")
    df["MAP_filled"] = df["MAP"].ffill().bfill()

    # resample to 1s for stable scoring
    df1 = _resample_to_1s(df[["time", "MAP_filled"]])
    s = df1["MAP_filled"].astype(np.float64)

    # Slopes
    slope_10 = (s - s.shift(10)) / 10.0
    slope_60 = (s - s.shift(60)) / 60.0

    # Variability
    std_30 = s.rolling(30, min_periods=10).std()
    absdiff_1 = s.diff().abs().rolling(30, min_periods=10).mean()

    # Scores (normalize-like)
    score_A = (-slope_10).clip(lower=0)          # rapid drops
    score_B = (-slope_60).clip(lower=0)          # gradual sustained
    score_C = (std_30.fillna(0) + absdiff_1.fillna(0))  # intermittent/noisy drops

    out = pd.DataFrame({
        "time": df1["time"].astype(np.float64),
        "score_A": score_A.fillna(0).astype(np.float64),
        "score_B": score_B.fillna(0).astype(np.float64),
        "score_C": score_C.fillna(0).astype(np.float64),
    })
    return out


def choose_drop_type(scores_df: pd.DataFrame) -> str:
    """
    Pick A/B/C based on average score (can be improved later).
    """
    meanA = float(scores_df["score_A"].mean())
    meanB = float(scores_df["score_B"].mean())
    meanC = float(scores_df["score_C"].mean())

    m = max(meanA, meanB, meanC)
    if m == meanA:
        return "A"
    if m == meanB:
        return "B"
    return "C"
