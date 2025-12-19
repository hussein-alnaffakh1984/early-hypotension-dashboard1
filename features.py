import pandas as pd

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature dataframe exactly as used during training
    Required inputs: MAP, HR, SpO2, (optional RR)
    """

    X = pd.DataFrame(index=df.index)

    X["MAP"] = df["MAP"]
    X["HR"] = df["HR"]
    X["SpO2"] = df["SpO2"]

    if "RR" in df.columns:
        X["RR"] = df["RR"]
    else:
        X["RR"] = 16  # default normal RR

    # simple temporal features (safe)
    X["MAP_diff"] = df["MAP"].diff().fillna(0)
    X["HR_diff"] = df["HR"].diff().fillna(0)

    return X
