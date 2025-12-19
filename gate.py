import numpy as np
import pandas as pd

def apply_gate(df_vitals: pd.DataFrame) -> pd.Series:
    """
    يرجع gate_mask (True = اسمح بالإنذار/التنبؤ)
    Gate بسيط:
      - إذا MAP منخفض جدًا (<= 65) -> True
      - أو إذا MAP_drop_2m <= -8 -> True
      - غير ذلك False
    """
    # افتراضات آمنة لو الأعمدة ناقصة
    map_now = pd.to_numeric(df_vitals.get("MAP", np.nan), errors="coerce")
    map_drop_2m = pd.to_numeric(df_vitals.get("MAP_drop_2m", np.nan), errors="coerce")

    cond1 = map_now <= 65
    cond2 = map_drop_2m <= -8

    mask = (cond1 | cond2).fillna(False)
    return mask.astype(bool)
