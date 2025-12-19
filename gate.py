import pandas as pd
import numpy as np

def apply_gate(X: pd.DataFrame) -> pd.DataFrame:
    """
    Gate بسيط: يمنع التنبؤ عندما MAP غير منطقي أو SpO2/HR مفقود.
    بدل ما نحذف الصفوف (يعقد dashboard)، نخليها لكن نخفّض تأثيرها:
    - أي صف غير موثوق نخليه صف صفري (features=0)
    """
    Y = X.copy()

    # define "valid" rows
    valid = pd.Series(True, index=Y.index)

    if "MAP" in Y.columns:
        valid &= (Y["MAP"] > 20) & (Y["MAP"] < 220)
    if "HR" in Y.columns:
        valid &= (Y["HR"] > 20) & (Y["HR"] < 250)
    if "SpO2" in Y.columns:
        valid &= (Y["SpO2"] > 30) & (Y["SpO2"] <= 100)

    # set invalid rows to zero (safe for the model)
    Y.loc[~valid, :] = 0.0
    Y = Y.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return Y
