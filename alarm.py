def generate_alarm(risk_score: float, threshold: float) -> int:
    return int(risk_score >= threshold)
