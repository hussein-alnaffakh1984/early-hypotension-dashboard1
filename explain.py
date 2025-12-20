# explain.py
import numpy as np


def build_medical_explanation(df_out, threshold: float, drop_key: str, use_gate: bool, lang: str = "en"):
    latest = df_out.iloc[-1]
    MAP = float(latest.get("MAP", np.nan))
    HR = float(latest.get("HR", np.nan))
    SpO2 = float(latest.get("SpO2", np.nan))
    RR = float(latest.get("RR", np.nan)) if "RR" in df_out.columns else np.nan
    risk = float(latest.get("risk_score", 0.0))
    alarm = bool(latest.get("alarm", False))

    if lang == "ar":
        reasons_title = "لماذا؟"
        rec_title = "التوصيات"
        disclaimer = "تنبيه: هذا النظام مساعد قرار وليس بديلاً عن التقييم الطبي."
    else:
        reasons_title = "Why?"
        rec_title = "Recommendation"
        disclaimer = "Disclaimer: This is a decision-support tool and does not replace clinical judgment."

    # Headline
    if lang == "ar":
        headline = "🚨 إنذار مبكر: خطر هبوط ضغط" if alarm else "✅ لا يوجد إنذار حاليًا"
    else:
        headline = "🚨 Early Warning: Hypotension Risk" if alarm else "✅ No alarm at this moment"

    reasons = []
    rec = []

    # Reasons
    if not np.isnan(MAP):
        if MAP < 65:
            reasons.append(("MAP أقل من 65 mmHg" if lang == "ar" else "MAP is below 65 mmHg (hypotension threshold)."))
        else:
            reasons.append(("MAP ضمن المجال المقبول" if lang == "ar" else "MAP is within an acceptable range."))

    if not np.isnan(HR):
        if HR > 100:
            reasons.append(("HR مرتفع (تسرّع قلبي تعويضي محتمل)" if lang == "ar" else "HR is elevated (possible compensatory tachycardia)."))

    if not np.isnan(SpO2):
        if SpO2 < 92:
            reasons.append(("SpO2 منخفض (<92%)" if lang == "ar" else "SpO2 is low (<92%)."))

    if not np.isnan(RR):
        if RR > 24:
            reasons.append(("RR مرتفع (>24)" if lang == "ar" else "RR is elevated (>24)."))

    # Model logic
    if risk >= threshold:
        reasons.append((f"درجة الخطر {risk:.3f} ≥ العتبة {threshold:.2f}" if lang == "ar" else f"Risk score {risk:.3f} ≥ threshold {threshold:.2f}."))
    else:
        reasons.append((f"درجة الخطر {risk:.3f} < العتبة {threshold:.2f}" if lang == "ar" else f"Risk score {risk:.3f} < threshold {threshold:.2f}."))

    # Drop type
    if lang == "ar":
        reasons.append(f"نمط الهبوط المختار: {drop_key}")
    else:
        reasons.append(f"Selected drop pattern mode: {drop_key}")

    if use_gate:
        reasons.append(("Gate مفعّل (تركيز على النمط المختار)" if lang == "ar" else "Gate enabled (pattern-focused selection)."))

    # Recommendations
    if alarm:
        if lang == "ar":
            rec = [
                "راجع ضغط المريض فورًا وتأكد من قراءة MAP.",
                "افحص السبب (نزف/تخدير/سوائل/أدوية موسعة).",
                "فكر بإجراءات دعم الدورة الدموية حسب البروتوكول.",
                "راقب التطور خلال الدقائق القادمة."
            ]
        else:
            rec = [
                "Re-check MAP immediately and confirm measurement quality.",
                "Assess potential causes (bleeding/anesthesia/fluids/vasodilation).",
                "Consider hemodynamic support per local protocol.",
                "Monitor trend closely over the next minutes."
            ]
    else:
        if lang == "ar":
            rec = [
                "استمر بالمراقبة.",
                "إذا ظهرت أعراض أو بدأ MAP ينخفض بسرعة، فعّل الإنذار بعتبة أقل أو راجع الإعدادات."
            ]
        else:
            rec = [
                "Continue monitoring.",
                "If symptoms appear or MAP starts dropping rapidly, consider a lower threshold or review settings."
            ]

    return {
        "headline": headline,
        "reasons_title": reasons_title,
        "rec_title": rec_title,
        "reasons": reasons,
        "recommendation": rec,
        "disclaimer": disclaimer
    }
