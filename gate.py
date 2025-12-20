import numpy as np
import pandas as pd


def apply_gate(X: pd.DataFrame, drop_key: str = "A"):
    """
    Always returns: (X_out, gate_mask)

    gate_mask is a boolean array (len = rows of X)
    drop_key:
      A = Rapid
      B = Gradual
      C = Intermittent
    """

    X_out = X.copy()

    # If MAP_drop_2m exists, we can use it as a gate signal.
    if "MAP_drop_2m" not in X_out.columns:
        gate_mask = np.ones(len(X_out), dtype=bool)
        return X_out, gate_mask

    d = X_out["MAP_drop_2m"].to_numpy()

    # Gate rules (simple, but stable):
    if drop_key == "A":
        # rapid: sharp negative drop
        gate_mask = d <= -10
    elif drop_key == "B":
        # gradual: mild drop sustained
        gate_mask = d <= -5
    else:
        # intermittent: fluctuating drops (abs changes)
        gate_mask = np.abs(d) >= 7

    # If mask is too strict (all False), don't block everything
    if gate_mask.sum() == 0:
        gate_mask = np.ones(len(X_out), dtype=bool)

    return X_out, gate_mask
