# drop_type.py
import numpy as np
import pandas as pd

def classify_drop_type(
    df: pd.DataFrame,
    time_col: str = "time",
    map_col: str = "MAP",
    rapid_window_min: float = 3.0,
    gradual_window_min: float = 15.0,
    rapid_drop_mmHg: float = 15.0,
    gradual_drop_mmHg: float = 20.0,
    intermittent_changes_min: int = 3,
):
    """
    Auto classify MAP drop pattern:
      A: Rapid        (big drop in short window)
      B: Gradual      (drop over longer window)
      C: Intermittent (oscillation / direction changes)

    Returns:
      {drop_key, label, details}
    """
    df = df.copy()
    if time_col not in df.columns or map_col not in df.columns:
        return {"drop_key": "B", "label": "B: Gradual", "details": {"reason": "Missing time/MAP"}}

    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
    m = pd.to_numeric(df[map_col], errors="coerce").to_numpy()

    mask = np.isfinite(t) & np.isfinite(m)
    t = t[mask]
    m = m[mask]

    if len(t) < 6:
        return {"drop_key": "B", "label": "B: Gradual", "details": {"reason": "Too few points", "n": int(len(t))}}

    order = np.argsort(t)
    t = t[order]
    m = m[order]

    # detect seconds vs minutes
    t_range = float(t[-1] - t[0])
    is_seconds = t_range > 120.0
    t_min = (t - t[0]) / 60.0 if is_seconds else (t - t[0])

    # smooth a bit
    m_s = pd.Series(m).rolling(3, center=True, min_periods=1).median().to_numpy()

    # slope
    dt = np.diff(t_min)
    dm = np.diff(m_s)
    slope = np.divide(dm, dt, out=np.zeros_like(dm, dtype=float), where=dt > 1e-9)

    # rapid drop within rapid_window
    rapid_drop = 0.0
    for i in range(len(t_min)):
        j = np.searchsorted(t_min, t_min[i] + rapid_window_min, side="right") - 1
        if j > i:
            drop = m_s[i] - np.min(m_s[i:j+1])
            rapid_drop = max(rapid_drop, float(drop))

    # gradual drop within gradual_window
    gradual_drop = 0.0
    for i in range(len(t_min)):
        j = np.searchsorted(t_min, t_min[i] + gradual_window_min, side="right") - 1
        if j > i:
            drop = m_s[i] - np.min(m_s[i:j+1])
            gradual_drop = max(gradual_drop, float(drop))

    # intermittent: direction changes
    s = slope.copy()
    s[np.abs(s) < 0.2] = 0.0
    sign = np.sign(s)
    sign_nz = sign[sign != 0]
    changes = int(np.sum(sign_nz[1:] * sign_nz[:-1] < 0)) if len(sign_nz) > 1 else 0

    details = {
        "time_unit": "seconds->minutes" if is_seconds else "minutes",
        "n_points": int(len(t_min)),
        "rapid_drop_mmHg": rapid_drop,
        "gradual_drop_mmHg": gradual_drop,
        "direction_changes": changes,
        "thresholds": {
            "rapid_window_min": rapid_window_min,
            "gradual_window_min": gradual_window_min,
            "rapid_drop_mmHg": rapid_drop_mmHg,
            "gradual_drop_mmHg": gradual_drop_mmHg,
            "intermittent_changes_min": intermittent_changes_min,
        },
    }

    # decide
    if changes >= intermittent_changes_min:
        return {"drop_key": "C", "label": "C: Intermittent", "details": details}
    if rapid_drop >= rapid_drop_mmHg:
        return {"drop_key": "A", "label": "A: Rapid", "details": details}
    if gradual_drop >= gradual_drop_mmHg:
        return {"drop_key": "B", "label": "B: Gradual", "details": details}

    return {"drop_key": "B", "label": "B: Gradual", "details": details}
