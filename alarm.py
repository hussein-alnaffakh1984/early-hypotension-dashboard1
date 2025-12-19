import numpy as np
import pandas as pd

def alarm_from_score(score: float, threshold: float) -> bool:
    return float(score) >= float(threshold)

def classify_drop_type(df: pd.DataFrame, map_col="MAP") -> str:
    """
    تصنيف بسيط من سلسلة MAP:
    A Rapid: هبوط سريع (فرق كبير خلال نافذة قصيرة)
    B Gradual: هبوط تدريجي (انحدار مستمر)
    C Intermittent: متقطع (ذبذبة/نمط صعود وهبوط)
    """
    if map_col not in df.columns or len(df) < 6:
        return "Unknown"

    x = pd.to_numeric(df[map_col], errors="coerce").fillna(method="ffill").fillna(0.0).to_numpy()

    # compute rough changes
    d1 = np.diff(x)
    if len(d1) == 0:
        return "Unknown"

    # rapid drop: any big negative jump
    if np.min(d1) <= -8:
        return "A: Rapid"

    # gradual: negative trend overall
    trend = np.polyfit(np.arange(len(x)), x, 1)[0]  # slope
    if trend < -0.05:
        return "B: Gradual"

    # intermittent: oscillation (many sign changes in derivative)
    sign_changes = np.sum(np.sign(d1[:-1]) != np.sign(d1[1:]))
    if sign_changes >= max(3, len(d1)//6):
        return "C: Intermittent"

    return "B: Gradual"
