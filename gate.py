import numpy as np
import pandas as pd

def apply_gate(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns (X_gated, gate_mask)
    gate_mask=True means row is OK, False means row was gated heavily.
    """
    X = X.copy()

    # row quality: if too many NaNs, mark as bad
    nan_ratio = X.isna().mean(axis=1)

    # basic clipping for numeric stability (keep NaN)
    X = X.replace([np.inf, -np.inf], np.nan)

    gate_mask = nan_ratio <= 0.60  # allow up to 60% NaN (model has imputer anyway)
    return X, gate_mask
