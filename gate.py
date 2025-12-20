# gate.py
import numpy as np
import pandas as pd


def apply_gate(X: pd.DataFrame, drop_key: str = "A"):
    """
    Gate makes A/B/C DIFFERENT by selecting rows based on pattern-relevant features.
    Returns (X_gated, mask).
    IMPORTANT: X shape stays same (we don't drop rows), we just build a mask.
    """
    X = X.copy()

    # default: keep all
    mask = pd.Series(True, index=X.index)

    # if needed columns missing, fall back to keep all
    def has(cols):
        return all(c in X.columns for c in cols)

    # A: Rapid -> emphasize abrupt drop in 2m and negative short-term change
    if drop_key == "A" and has(["MAP_drop_2m", "MAP_d60"]):
        score = (
            X["MAP_drop_2m"].fillna(0).clip(lower=0) +
            (-X["MAP_d60"].fillna(0).clip(upper=0))
        )
        thr = np.nanpercentile(score.to_numpy(), 60)  # keep top 40% most "rapid-like"
        mask = score >= thr

    # B: Gradual -> sustained long-term decline, not necessarily abrupt
    elif drop_key == "B" and has(["MAP_d60", "MAP_m60"]):
        # negative slope magnitude over 60 sec
        score = (-X["MAP_d60"].fillna(0).clip(upper=0))
        thr = np.nanpercentile(score.to_numpy(), 55)
        mask = score >= thr

    # C: Intermittent -> unstable / fluctuating (std)
    elif drop_key == "C" and has(["MAP_s60"]):
        score = X["MAP_s60"].fillna(0)
        thr = np.nanpercentile(score.to_numpy(), 60)
        mask = score >= thr

    # Return X unchanged + mask
    return X, mask
