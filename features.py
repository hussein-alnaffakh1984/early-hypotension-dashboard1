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
    return [
        "MAP_filled",
        "MAP_m30","MAP_m60","MAP_s60","MAP_d1","MAP_d60","MAP_drop_2m",
        "HR","HR_m30","HR_m60","HR_s60","HR_d1","HR_d60",
        "SpO2","SpO2_m30","SpO2_m60","SpO2_s60","SpO2_d1","SpO2_d60",
        "EtCO2","EtCO2_m30","EtCO2_m60","EtCO2_s60","EtCO2_d1","EtCO2_d60",
        "RR","RR_m30","RR_m60","RR_s60","RR_d1","RR_d60",
    ]


def _ensure_time_seconds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "time" not in df.columns:
        raise ValueError("CSV must contain 'time' column.")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def _resample_to_1s(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    if "time" not in df.columns:
        raise ValueError("Missing required column: time")

    # 1) time -> numeric
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    # 2) ✅ حول الوقت إلى ثواني INT (هذا يحل float/int نهائياً)
    # إذا الوقت أصلاً بالثواني وفيه كسور، نقرّبه لأقرب ثانية
    df["time_s"] = np.round(df["time"].astype(float)).astype(np.int64)

    # 3) إزالة التكرار بنفس الثانية (احتفظ بآخر قراءة)
    df = df.drop_duplicates(subset=["time_s"], keep="last").reset_index(drop=True)

    # 4) شبكة 1 ثانية INT أيضاً
    t_min = int(df["time_s"].min())
    t_max = int(df["time_s"].max())
    grid = pd.DataFrame({"time_s": np.arange(t_min, t_max + 1, 1, dtype=np.int64)})

    # 5) ✅ merge_asof على time_s (int مع int)
    df2 = pd.merge_asof(
        grid.sort_values("time_s"),
        df.sort_values("time_s"),
        on="time_s",
        direction="nearest",
        tolerance=0  # لأن time_s صار integer بالضبط
    )

    # 6) رجّع عمود time الأصلي (اختياري) أو اصنع time من time_s
    # نخلي time = time_s حتى يكون واضح أنه بالثواني
    df2["time"] = df2["time_s"].astype(np.float64)

    # احذف time_s إذا لا تحتاجه
    # (إذا عندك أجزاء أخرى تعتمد على time_s اتركه)
    return df2




    grid = pd.DataFrame({"time": np.arange(t0, t1 + 1, 1, dtype=float)})

    # asof merge to nearest previous sample then interpolate
    df2 = pd.merge_asof(
        grid.sort_values("time"),
        df.sort_values("time"),
        on="time",
        direction="backward",
        tolerance=10_000  # large tolerance
    )

    # interpolate for smoother time series
    for col in ["MAP", "HR", "SpO2", "RR", "EtCO2"]:
        df2[col] = df2[col].interpolate(limit_direction="both")

    return df2


def _rolling_feats(s: pd.Series, win: int):
    """
    win is in seconds on 1s grid.
    """
    m = s.rolling(win, min_periods=max(3, win // 3)).mean()
    std = s.rolling(win, min_periods=max(3, win // 3)).std()
    return m, std


def _diff_lag(s: pd.Series, lag: int):
    """
    Difference with lag seconds: current - lagged
    """
    return s - s.shift(lag)


def compute_drop_scores(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Produce per-time scores for Rapid/Gradual/Intermittent patterns.
    This is used to make A/B/C actually DIFFERENT (AUTO + weighting).
    """
    df = _resample_to_1s(df_raw)

    map_f = df["MAP"].ffill().bfill()
    # rapid: sharp drop over 2 minutes + negative slope
    drop_2m = (map_f.shift(120) - map_f).clip(lower=0)  # positive when dropping
    slope_60 = (map_f - map_f.shift(60))  # negative means falling

    rapid_score = (drop_2m / 15.0) + (-slope_60.clip(upper=0) / 10.0)

    # gradual: sustained fall over 60-180 sec (without big abrupt)
    drop_1m = (map_f.shift(60) - map_f).clip(lower=0)
    drop_3m = (map_f.shift(180) - map_f).clip(lower=0)
    gradual_score = (drop_3m / 20.0) + (drop_1m / 15.0) - (drop_2m / 20.0)

    # intermittent: oscillation/instability (variance + sign changes)
    _, std_60 = _rolling_feats(map_f, 60)
    d1 = map_f.diff()
    sign_changes = (np.sign(d1).diff().abs() > 0).astype(float).rolling(60, min_periods=10).mean()
    intermittent_score = (std_60.fillna(0) / 6.0) + (sign_changes.fillna(0) / 0.25)

    scores = pd.DataFrame({
        "time": df["time"].values,
        "score_A_rapid": rapid_score.replace([np.inf, -np.inf], np.nan).fillna(0).values,
        "score_B_gradual": gradual_score.replace([np.inf, -np.inf], np.nan).fillna(0).values,
        "score_C_intermittent": intermittent_score.replace([np.inf, -np.inf], np.nan).fillna(0).values,
    })

    # normalize 0..1 per case
    for c in ["score_A_rapid", "score_B_gradual", "score_C_intermittent"]:
        mx = float(scores[c].max()) if len(scores) else 0.0
        if mx <= 1e-9:
            scores[c + "_n"] = 0.0
        else:
            scores[c + "_n"] = (scores[c] / mx).clip(0, 1)

    # AUTO label per time
    arr = scores[["score_A_rapid_n", "score_B_gradual_n", "score_C_intermittent_n"]].to_numpy()
    idx = np.argmax(arr, axis=1)
    labels = np.array(["A", "B", "C"])[idx]
    scores["drop_auto"] = labels

    return scores


def build_feature_matrix(df_raw: pd.DataFrame, expected_cols: list) -> pd.DataFrame:
    """
    Build X EXACTLY with the columns the model expects.
    Uses 1-second resampling and window-based rolling features.
    """
    df = _resample_to_1s(df_raw)

    # signals
    MAP = df["MAP"].copy()
    HR = df["HR"].copy()
    SpO2 = df["SpO2"].copy()
    RR = df["RR"].copy()
    EtCO2 = df["EtCO2"].copy()

    MAP_filled = MAP.ffill().bfill()

    # rolling windows
    MAP_m30, _ = _rolling_feats(MAP_filled, 30)
    MAP_m60, MAP_s60 = _rolling_feats(MAP_filled, 60)
    HR_m30, _ = _rolling_feats(HR, 30)
    HR_m60, HR_s60 = _rolling_feats(HR, 60)
    Sp_m30, _ = _rolling_feats(SpO2, 30)
    Sp_m60, Sp_s60 = _rolling_feats(SpO2, 60)
    Et_m30, _ = _rolling_feats(EtCO2, 30)
    Et_m60, Et_s60 = _rolling_feats(EtCO2, 60)
    RR_m30, _ = _rolling_feats(RR, 30)
    RR_m60, RR_s60 = _rolling_feats(RR, 60)

    # diffs
    MAP_d1 = _diff_lag(MAP_filled, 1)
    MAP_d60 = _diff_lag(MAP_filled, 60)
    MAP_drop_2m = (MAP_filled.shift(120) - MAP_filled).clip(lower=0)

    HR_d1 = _diff_lag(HR, 1)
    HR_d60 = _diff_lag(HR, 60)

    Sp_d1 = _diff_lag(SpO2, 1)
    Sp_d60 = _diff_lag(SpO2, 60)

    Et_d1 = _diff_lag(EtCO2, 1)
    Et_d60 = _diff_lag(EtCO2, 60)

    RR_d1 = _diff_lag(RR, 1)
    RR_d60 = _diff_lag(RR, 60)

    # assemble base df with ALL possible features we might need
    feat = pd.DataFrame({
        "MAP_filled": MAP_filled,
        "MAP_m30": MAP_m30, "MAP_m60": MAP_m60, "MAP_s60": MAP_s60,
        "MAP_d1": MAP_d1, "MAP_d60": MAP_d60, "MAP_drop_2m": MAP_drop_2m,

        "HR": HR,
        "HR_m30": HR_m30, "HR_m60": HR_m60, "HR_s60": HR_s60,
        "HR_d1": HR_d1, "HR_d60": HR_d60,

        "SpO2": SpO2,
        "SpO2_m30": Sp_m30, "SpO2_m60": Sp_m60, "SpO2_s60": Sp_s60,
        "SpO2_d1": Sp_d1, "SpO2_d60": Sp_d60,

        "EtCO2": EtCO2,
        "EtCO2_m30": Et_m30, "EtCO2_m60": Et_m60, "EtCO2_s60": Et_s60,
        "EtCO2_d1": Et_d1, "EtCO2_d60": Et_d60,

        "RR": RR,
        "RR_m30": RR_m30, "RR_m60": RR_m60, "RR_s60": RR_s60,
        "RR_d1": RR_d1, "RR_d60": RR_d60,
    })

    # numeric cleanup
    feat = feat.replace([np.inf, -np.inf], np.nan)

    # return ONLY expected columns in exact order (missing -> NaN)
    X = feat.reindex(columns=list(expected_cols), fill_value=np.nan)
    return X
