# explain.py
import numpy as np


def _t(lang: str, en: str, ar: str) -> str:
    return en if lang == "en" else ar


def build_medical_explanation(df_out, threshold: float, drop_key: str, use_gate: bool, lang: str = "en"):
    """
    Builds a safe, non-diagnostic explanation text for the dashboard & PDF.
    """
    if df_out is None or len(df_out) == 0:
        return {
            "headline": _t(lang, "No data available.", "لا توجد بيانات متاحة."),
            "reasons_title": _t(lang, "Why?", "لماذا؟"),
            "rec_title": _t(lang, "Recommendation", "التوصيات"),
            "reasons": [],
            "recommendation": [],
            "disclaimer": _t(
                lang,
                "Disclaimer: This tool is for decision support only and does not replace clinical judgment.",
                "تنبيه: هذا النظام للمساعدة على اتخاذ القرار ولا يغني عن الحكم السريري."
            )
        }

    last = df_out.iloc[-1]
    MAP = float(last.get("MAP", np.nan))
    HR = float(last.get("HR", np.nan))
    SpO2 = float(last.get("SpO2", np.nan))
    risk = float(last.get("risk_score", np.nan))
    alarm = bool(last.get("alarm", False))

    reasons = []
    recs = []

    # Headline
    if alarm:
        headline = _t(
            lang,
            f"ALERT: Elevated risk of hypotension (risk={risk:.3f} ≥ {threshold:.2f}).",
            f"إنذار: خطر مرتفع لهبوط الضغط (الخطر={risk:.3f} ≥ {threshold:.2f})."
        )
    else:
        headline = _t(
            lang,
            f"No alert: Risk below threshold (risk={risk:.3f} < {threshold:.2f}).",
            f"لا يوجد إنذار: الخطر أقل من العتبة (الخطر={risk:.3f} < {threshold:.2f})."
        )

    # Reasons (explainable)
    if np.isfinite(MAP):
        if MAP < 65:
            reasons.append(_t(lang, f"MAP is low ({MAP:.1f} mmHg).", f"MAP منخفض ({MAP:.1f} mmHg)."))
        else:
            reasons.append(_t(lang, f"MAP is {MAP:.1f} mmHg.", f"MAP = {MAP:.1f} mmHg."))

    if np.isfinite(HR):
        if HR > 100:
            reasons.append(_t(lang, f"HR is elevated ({HR:.0f} bpm).", f"HR مرتفع ({HR:.0f} bpm)."))
        else:
            reasons.append(_t(lang, f"HR is {HR:.0f} bpm.", f"HR = {HR:.0f} bpm."))

    if np.isfinite(SpO2):
        if SpO2 < 92:
            reasons.append(_t(lang, f"SpO2 is low ({SpO2:.0f}%).", f"SpO2 منخفض ({SpO2:.0f}%)."))
        else:
            reasons.append(_t(lang, f"SpO2 is {SpO2:.0f}%.", f"SpO2 = {SpO2:.0f}%."))

    reasons.append(_t(lang, f"Drop type used: {drop_key}.", f"نوع الهبوط المستخدم: {drop_key}."))

    if use_gate:
        reasons.append(_t(lang, "Gate is enabled (risk may be suppressed outside the pattern).",
                          "التصفية (Gate) مفعلة (قد يتم تقليل الخطر خارج النمط)."))

    # Recommendations (safe)
    if alarm:
        recs.extend([
            _t(lang, "Re-check MAP and validate sensor/line readings.", "أعد قياس MAP وتحقق من دقة الحساس/الخط."),
            _t(lang, "Assess volume status and clinical context.", "قيّم حالة السوائل والسياق السريري."),
            _t(lang, "Follow local ICU/OR hypotension protocol if clinically indicated.", "اتبع بروتوكول هبوط الضغط في القسم إذا استدعى الأمر سريريًا."),
        ])
    else:
        recs.extend([
            _t(lang, "Continue routine monitoring.", "استمر بالمراقبة الروتينية."),
            _t(lang, "If vitals trend downward, consider closer observation.", "إذا كانت المؤشرات تتجه للأسوأ، زد وتيرة المراقبة."),
        ])

    disclaimer = _t(
        lang,
        "Disclaimer: This tool is for decision support only and does not replace clinical judgment.",
        "تنبيه: هذا النظام للمساعدة على اتخاذ القرار ولا يغني عن الحكم السريري."
    )

    return {
        "headline": headline,
        "reasons_title": _t(lang, "Why?", "لماذا؟"),
        "rec_title": _t(lang, "Recommendation", "التوصيات"),
        "reasons": reasons,
        "recommendation": recs,
        "disclaimer": disclaimer
    }
