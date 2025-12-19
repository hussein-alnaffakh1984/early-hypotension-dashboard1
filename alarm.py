# alarm.py
def generate_alarm(risk_score: float, threshold: float) -> bool:
    return bool(risk_score >= threshold)
