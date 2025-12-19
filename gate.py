import pandas as pd
import numpy as np

def apply_gate(X: pd.DataFrame,
               drop_thr: float = -5.0,
               map_low_thr: float = 70.0) -> pd.DataFrame:
    """
    Expects X has MAP_drop_2m and/or MAP_m30 or MAP_m60.
    Returns filtered dataframe (only gated rows).
    """
    df = X.copy()

    # If some columns not present, just return X (no gate)
    has_drop = "MAP_drop_2m" in df.columns
    has_map_m30 = "MAP_m30" in df.columns
    has_map_m60 = "MAP_m60" in df.columns

    if not (has_drop or has_map_m30 or has_map_m60):
        return df

    gate_mask = pd.Series(False, index=df.index)

    if has_drop:
        gate_mask = gate_mask | (df["MAP_drop_2m"].astype(float) <= drop_thr)

    if has_map_m30:
        gate_mask = gate_mask | (df["MAP_m30"].astype(float) <= map_low_thr)

    if has_map_m60:
        gate_mask = gate_mask | (df["MAP_m60"].astype(float) <= map_low_thr)

    gated = df.loc[gate_mask].copy()
    # لو gate فلتر كل شيء بالغلط، نرجع آخر صف حتى لا ينهار التطبيق
    if len(gated) == 0:
        return df.tail(1).copy()
    return gated
