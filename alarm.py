def generate_alarm(risk_score: float, threshold: float) -> bool:
    try:
        return float(risk_score) >= float(threshold)
    except Exception:
        return False
