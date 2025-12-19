import numpy as np
import pandas as pd

def apply_gate(df_raw: pd.DataFrame, X: pd.DataFrame, drop_thr: float = -5.0, map_thr: float = 75.0):
    """
    Gate بسيط وقابل للتعديل:
    - إذا MAP_drop_2m <= drop_thr  (يعني هبوط خلال دقيقتين)
    OR
    - MAP_m60 <= map_thr          (قرب من hypotension)
    يرجع: mask (True/False لكل صف)
    """
    if "MAP_drop_2m" not in X.columns or "MAP_m60" not in X.columns:
        mask = np.ones(len(X), dtype=bool)
        return mask

    drop_ok = (X["MAP_drop_2m"].astype(float) <= float(drop_thr))
    map_ok  = (X["MAP_m60"].astype(float) <= float(map_thr))
    mask = (drop_ok | map_ok).to_numpy(dtype=bool)
    return mask
