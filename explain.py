# explain.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _fmt(x, nd=1):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def build_medical_explanation(df_out: pd.DataFrame, threshold: float, drop_key: str, use_gate: bool) -> dict:
    """
    Returns a dict:
      - headline: short status line
      - reasons: list[str] key reasons
      - recommendation: list[str] practical suggestions
      - disclaimer: str
    Expects df_out contains columns: time, MAP, HR, SpO2, (RR optional), risk_score, alarm
    """
    last = df_out.iloc[-1].copy()

    # Basic values
    last_map = float(last.get("MAP", np.nan))
    last_hr  = float(last.get("HR", np.nan))
    last_spo2 = float(last.get("SpO2", np.nan))
    last_rr = float(last.get("RR", np.nan)) if "RR" in df_out.columns else np.nan
    last_risk = float(last.get("risk_score", np.nan))
    last_alarm = bool(last.get("alarm", False))

    # Trends (first -> last)
    first = df_out.iloc[0]
    map_drop = float(first.get("MAP", np.nan)) - last_map
    hr_rise  = last_hr - float(first.get("HR", np.nan))
    spo2_drop = float(first.get("SpO2", np.nan)) - last_spo2
    rr_rise = (last_rr - float(first.get("RR", np.nan))) if ("RR" in df_out.columns and "RR" in first) else np.nan

    # Simple clinical flags
    flags = []
    if np.isfinite(last_map) and last_map < 65:
        flags.append(f"MAP منخفض (<65): الحالي {_fmt(last_map)} ممHg")
    if np.isfinite(last_hr) and last_hr >= 100:
        flags.append(f"تسرّع قلب (≥100): الحالي {_fmt(last_hr)} bpm")
    if np.isfinite(last_spo2) and last_spo2 < 92:
        flags.append(f"SpO₂ منخفض (<92%): الحالي {_fmt(last_spo2)}%")
    if np.isfinite(last_rr) and last_rr >= 22:
        flags.append(f"تسرّع تنفّس (RR ≥22): الحالي {_fmt(last_rr)} /min")

    trend_msgs = []
    if np.isfinite(map_drop) and map_drop > 10:
        trend_msgs.append(f"انخفاض MAP بمقدار ~{_fmt(map_drop)} ممHg خلال السلسلة")
    if np.isfinite(hr_rise) and hr_rise > 10:
        trend_msgs.append(f"ارتفاع HR بمقدار ~{_fmt(hr_rise)} bpm (استجابة تعويضية محتملة)")
    if np.isfinite(spo2_drop) and spo2_drop > 3:
        trend_msgs.append(f"انخفاض SpO₂ بمقدار ~{_fmt(spo2_drop)}%")

    # Drop type text
    drop_map = {
        "A": "A (Rapid) هبوط سريع",
        "B": "B (Gradual) هبوط تدريجي",
        "C": "C (Intermittent) هبوط متقطع",
    }
    drop_text = drop_map.get(drop_key, str(drop_key))

    # Headline
    if last_alarm:
        headline = f"🚨 إنذار هبوط ضغط محتمل (Risk={_fmt(last_risk,2)} ≥ Threshold={_fmt(threshold,2)})"
    else:
        headline = f"✅ لا يوجد إنذار حاليًا (Risk={_fmt(last_risk,2)} < Threshold={_fmt(threshold,2)})"

    reasons = []
    reasons.append(f"نوع الهبوط المختار: {drop_text}")
    if use_gate:
        reasons.append("Gate مفعّل: تم تطبيق فلترة/بوابة قبل التنبؤ (قد تقلل إنذارات كاذبة)")
    else:
        reasons.append("Gate غير مفعّل: التنبؤ مباشرة من الميزات المستخرجة")

    # Add key reasons
    if flags:
        reasons.extend(flags)
    if trend_msgs:
        reasons.extend(trend_msgs)

    # If no flags, still explain with risk
    if not flags and not trend_msgs:
        reasons.append("تم رفع درجة الخطر بناءً على نمط تغيّر العلامات الحيوية ضمن النافذة الزمنية.")

    # Recommendations (generic, safe)
    recommendation = []
    if last_alarm:
        recommendation += [
            "إعادة قياس ضغط الدم والتأكد من جودة الإشارة/الكُفّة/القياس.",
            "تقييم سريع لعلامات الصدمة ونقص التروية (وعي، بول، برودة الأطراف).",
            "النظر في إعطاء سوائل إذا كان مناسبًا سريريًا وتقييم الحاجة لمقبّضات وعائية حسب البروتوكول.",
            "فحص السبب المحتمل: نزف، إنتان، أدوية/تخدير، نقص حجم، اضطراب نظم.",
            "إذا كانت الحالة ICU/OR: إبلاغ الفريق المسؤول فورًا واتباع بروتوكول المستشفى."
        ]
    else:
        recommendation += [
            "استمر بالمراقبة الدورية للعلامات الحيوية.",
            "إذا ظهرت أعراض أو تغيّر سريع، خفّض Threshold أو فعّل Gate حسب الحاجة."
        ]

    disclaimer = (
        "هذا التفسير ناتج عن نموذج تعلم آلي لغرض الإنذار المبكر فقط "
        "ولا يُستخدم كتشخيص نهائي أو بديل عن قرار الطبيب."
    )

    return {
        "headline": headline,
        "reasons": reasons,
        "recommendation": recommendation,
        "disclaimer": disclaimer
    }
