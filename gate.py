import pandas as pd

def apply_gate(X: pd.DataFrame) -> pd.DataFrame:
    """
    Simple gate: keep all rows but add a gate flag column.
    (You can later replace with your real gate logic.)
    """
    X = X.copy()
    # Example gate: if MAP exists and is reasonable
    if "MAP_val" in X.columns:
        X["gate"] = (X["MAP_val"] > 20).astype(int)
    else:
        X["gate"] = 1
    return X
