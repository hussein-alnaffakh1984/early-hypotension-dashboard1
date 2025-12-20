import numpy as np
import pandas as pd


def get_expected_feature_columns(model):
    """
    Use the model's own expected feature names (most reliable).
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    # fallback (should not happen if model was trained with DataFrame)
    return ["MAP_filled", "HR", "SpO2", "RR", "EtCO2"]


def build_feature_matrix(df_raw: pd.DataFrame, expected_cols: list) -> pd.DataFrame:
    """
    Build X that matches EXACTLY the columns the model was trained on.
    """
    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]

    # Required base signals
    # time, MAP, HR, SpO2 required in CSV
    if "MAP" not in df.columns or "HR" not in df.columns or "SpO2" not in df.columns:
        raise ValueError("CSV must contain MAP, HR, SpO2 (and time).")

    # Optional signals
    if "RR" not in df.columns:
        df["RR"] = np.nan
    if "EtCO2" not in df.columns:
        df["EtCO2"] = np.nan

    # Create MAP_filled exactly as training name
    df["MAP_filled"] = pd.to_numeric(df["MAP"], errors="coerce").ffill().bfill()

    # Force numeric
    for c in ["MAP_filled", "HR", "SpO2", "RR", "EtCO2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Now build X ONLY with expected columns
    X = pd.DataFrame(index=df.index)
    for c in expected_cols:
        if c in df.columns:
            X[c] = df[c]
        else:
            # if model expects a column not available, create it safely
            X[c] = 0.0

    return X
