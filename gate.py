# gate.py
import pandas as pd

def apply_gate(X: pd.DataFrame) -> pd.DataFrame:
    """
    Simple gate: keep rows where MAP is valid and within plausible range.
    You can tighten/relax later.
    """
    if "MAP" not in X.columns:
        return X

    mask = (X["MAP"] >= 30) & (X["MAP"] <= 160)
    Xg = X.loc[mask].copy()

    # إذا كل شيء انحذف، رجّع الأصل حتى لا يصير فارغ
    if len(Xg) == 0:
        return X.copy()
    return Xg
