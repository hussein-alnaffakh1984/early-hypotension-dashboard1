import numpy as np

def refractory_alarm(risk: np.ndarray, thr: float = 0.15, refract_sec: float = 180.0, time_sec=None):
    """
    Alarm when risk>=thr with refractory: بعد ما يطلق إنذار، يمنع إنذارات ثانية لمدة refract_sec.
    """
    risk = np.asarray(risk, dtype=float)
    n = len(risk)
    alarm = np.zeros(n, dtype=int)

    if time_sec is None:
        # assume 1 sec steps
        time_sec = np.arange(n, dtype=float)
    else:
        time_sec = np.asarray(time_sec, dtype=float)

    last_alarm_t = -1e18
    for i in range(n):
        if risk[i] >= thr and (time_sec[i] - last_alarm_t) >= refract_sec:
            alarm[i] = 1
            last_alarm_t = time_sec[i]
    return alarm
