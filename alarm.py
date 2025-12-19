import numpy as np

def generate_alarm(risk_score: float, threshold: float) -> bool:
    try:
        return float(risk_score) >= float(threshold)
    except Exception:
        return False

def classify_drop_type(map_values: np.ndarray, time_values: np.ndarray) -> str:
    """
    Auto classify hypotension pattern:
    A: Rapid (steep drop in short time)
    B: Gradual (slow consistent downward trend)
    C: Intermittent (zig-zag / oscillation with drops)
    """
    y = np.asarray(map_values, dtype=float)
    t = np.asarray(time_values, dtype=float)

    if len(y) < 6:
        return "B"  # default

    # remove non-finite
    mask = np.isfinite(y) & np.isfinite(t)
    y = y[mask]
    t = t[mask]
    if len(y) < 6:
        return "B"

    # compute slopes
    dt = np.diff(t)
    dy = np.diff(y)
    dt[dt == 0] = 1e-6
    slope = dy / dt  # MAP units per second

    # recent window
    k = min(30, len(slope))
    s = slope[-k:]

    # Rapid: strong negative slope spike
    if np.min(s) <= -0.5:   # adjust if needed
        return "A"

    # Intermittent: many sign changes + noticeable drops
    sign_changes = np.sum(np.sign(s[1:]) != np.sign(s[:-1]))
    drop_mag = (np.max(y[-(k+1):]) - np.min(y[-(k+1):]))
    if sign_changes >= max(3, k // 5) and drop_mag >= 8:
        return "C"

    # Gradual: overall downward trend
    # linear fit slope
    x = t[-(k+1):]
    yy = y[-(k+1):]
    if len(x) >= 6:
        m = np.polyfit(x, yy, 1)[0]  # MAP per second
        if m < -0.05:
            return "B"

    return "B"
