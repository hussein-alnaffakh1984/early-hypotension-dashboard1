# explain.py
import numpy as np


def _safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _trend_last(df, col, n=10):
    """Return slope-like trend using last n points (simple diff / n)."""
    if col not in df.columns:
        return np.nan
    s = df[col].to_numpy()
    if len(s) < 2:
        return np.nan
    k = min(n, len(s) - 1)
    tail = s[-(k + 1):]
    tail = tail.astype(float)
    return (tail[-1] - tail[0]) / max(k, 1)


def build_medical_explanation(df_out, threshold=0.11, drop_key="A", use_gate=True, lang="en"):
    """
    Returns dict:
      headline, reasons (list), recommendation (list), disclaimer
      + optional localized titles: reasons_title, rec_title
    """
    latest = df_out.iloc[-1]

    MAP = _safe_float(latest.get("MAP", np.nan))
    HR = _safe_float(latest.get("HR", np.nan))
    SpO2 = _safe_float(latest.get("SpO2", np.nan))
    RR = _safe_float(latest.get("RR", np.nan))
    EtCO2 = _safe_float(latest.get("EtCO2", np.nan))
    risk = _safe_float(latest.get("risk_score", np.nan))

    # Trends (last points)
    map_tr = _trend_last(df_out, "MAP", n=10)
    hr_tr = _trend_last(df_out, "HR", n=10)
    spo2_tr = _trend_last(df_out, "SpO2", n=10)

    alarm_on = bool(risk >= threshold)

    # Localized strings
    if lang == "ar":
        reasons_title = "لماذا؟"
        rec_title = "التوصيات"
        disclaimer = "تنبيه: هذا النظام مساعد قرار ولا يغني عن التقييم الطبي السريري."
        if alarm_on:
            headline = "🚨 إنذار مبكر: خطر هبوط ضغط مرتفع خلال الفترة القريبة."
        else:
            headline = "✅ لا يوجد إنذار: خطر الهبوط منخفض حاليًا."
    else:
        reasons_title = "Why?"
        rec_title = "Recommendation"
        disclaimer = "Disclaimer: This is a decision-support tool and does not replace clinical judgment."
        if alarm_on:
            headline = "🚨 Early Warning: High near-term hypotension risk."
        else:
            headline = "✅ No Alarm: Low near-term hypotension risk."

    reasons = []
    recs = []

    # Reasons rules (simple, interpretable)
    # 1) MAP absolute
    if not np.isnan(MAP):
        if MAP < 65:
            reasons.append("MAP < 65 mmHg (hypotension range)." if lang == "en"
                           else "MAP أقل من 65 مم زئبق (ضمن نطاق هبوط الضغط).")
        elif MAP < 70:
            reasons.append("MAP borderline (65–70 mmHg)." if lang == "en"
                           else "MAP قريب من الحد (65–70 مم زئبق).")

    # 2) MAP trend
    if not np.isnan(map_tr):
        if map_tr <= -0.5:
            reasons.append("MAP is trending down (rapid decline)." if lang == "en"
                           else "MAP يتجه للانخفاض (هبوط سريع).")
        elif map_tr < 0:
            reasons.append("MAP is trending down." if lang == "en"
                           else "MAP يتجه للانخفاض.")

    # 3) HR
    if not np.isnan(HR):
        if HR > 100:
            reasons.append("HR > 100 bpm (possible compensatory tachycardia)." if lang == "en"
                           else "HR أعلى من 100/دقيقة (قد يكون تعويضًا/تسرع).")

    # 4) SpO2 trend/low
    if not np.isnan(SpO2):
        if SpO2 < 92:
            reasons.append("SpO2 < 92% (possible oxygenation issue)." if lang == "en"
                           else "SpO2 أقل من 92% (قد توجد مشكلة أكسجة).")
    if not np.isnan(spo2_tr) and spo2_tr < 0:
        reasons.append("SpO2 is decreasing." if lang == "en"
                       else "SpO2 يتناقص.")

    # 5) Model reason
    if not np.isnan(risk):
        reasons.append(
            (f"Model risk_score = {risk:.3f} vs threshold = {threshold:.2f}."
             if lang == "en"
             else f"قيمة الخطر من النموذج = {risk:.3f} مقابل العتبة = {threshold:.2f}.")
        )

    # 6) Gate
    if use_gate:
        reasons.append("Gate is enabled (alerts focus on clinically relevant segments)." if lang == "en"
                       else "Gate مُفعّل (يركّز الإنذار على المقاطع المهمة سريريًا).")

    # Drop type note
    drop_map = {"A": ("Rapid", "سريع"), "B": ("Gradual", "تدريجي"), "C": ("Intermittent", "متقطع")}
    if drop_key in drop_map:
        reasons.append(
            (f"Selected drop type: {drop_map[drop_key][0]}."
             if lang == "en"
             else f"نوع الهبوط المختار: {drop_map[drop_key][1]}.")
        )

    # Recommendations (generic safe)
    if alarm_on:
        if lang == "ar":
            recs.extend([
                "تحقق من القراءة والأجهزة (قياس الضغط/الخط/المستشعر).",
                "راجع الاتجاه خلال آخر دقائق (MAP/HR/SpO2) وتأكد من وجود هبوط فعلي.",
                "قيّم الحالة السريرية: علامات نقص التروية، نزف، عمق التخدير، السوائل.",
                "اتبع بروتوكول القسم للتعامل مع هبوط الضغط إن لزم."
            ])
        else:
            recs.extend([
                "Verify signal quality and device readings (BP line/cuff/sensors).",
                "Review recent trends (MAP/HR/SpO2) to confirm a true decline.",
                "Assess clinically: perfusion signs, bleeding, anesthetic depth, fluid status.",
                "Follow your unit protocol for hypotension management if confirmed."
            ])
    else:
        if lang == "ar":
            recs.extend([
                "استمر بالمراقبة الدورية.",
                "إذا ظهرت أعراض أو بدأ MAP بالانخفاض، أعد التقييم."
            ])
        else:
            recs.extend([
                "Continue routine monitoring.",
                "Reassess if symptoms appear or MAP starts to decline."
            ])

    # Ensure non-empty
    if not reasons:
        reasons = ["Insufficient data to explain." if lang == "en" else "لا توجد بيانات كافية للتفسير."]
    if not recs:
        recs = ["Monitor and reassess." if lang == "en" else "راقب وأعد التقييم."]

    return {
        "headline": headline,
        "reasons_title": reasons_title,
        "reasons": reasons,
        "rec_title": rec_title,
        "recommendation": recs,
        "disclaimer": disclaimer
    }
