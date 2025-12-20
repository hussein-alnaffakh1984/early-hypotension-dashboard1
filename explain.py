def build_medical_explanation(df_out, threshold: float, drop_key: str, use_gate: bool, lang: str = "en"):
    """
    Returns dict:
      headline, reasons(list), recommendation(list), disclaimer
    lang: "en" or "ar"
    """
    latest = df_out.iloc[-1]
    MAP = float(latest.get("MAP", float("nan")))
    HR = float(latest.get("HR", float("nan"))) if "HR" in df_out.columns else float("nan")
    SpO2 = float(latest.get("SpO2", float("nan"))) if "SpO2" in df_out.columns else float("nan")
    risk = float(latest.get("risk_score", float("nan")))
    alarm = bool(latest.get("alarm", False))

    # Basic trend hints
    map_prev = float(df_out.iloc[-2]["MAP"]) if len(df_out) > 1 else MAP
    map_drop = map_prev - MAP

    if lang == "ar":
        headline_alarm = "🚨 إنذار: خطر هبوط ضغط مرتفع"
        headline_ok = "✅ لا يوجد إنذار: الخطر منخفض"
        reasons_title = "الأسباب المحتملة:"
        rec_title = "التوصيات:"
        disclaimer = "تنبيه: هذا تفسير آلي مساعد وليس بديلاً عن قرار الطبيب."
    else:
        headline_alarm = "🚨 Alert: High risk of hypotension"
        headline_ok = "✅ No Alert: Low risk"
        reasons_title = "Possible reasons:"
        rec_title = "Recommendations:"
        disclaimer = "Disclaimer: Automated support only; not a substitute for clinician judgment."

    reasons = []
    recs = []

    # Reasons
    if not (MAP != MAP):  # not NaN
        if MAP < 65:
            reasons.append("MAP < 65 mmHg (hypotension range)." if lang == "en" else "MAP أقل من 65 ملم زئبق (نطاق هبوط الضغط).")
        if map_drop >= 10:
            reasons.append("Recent MAP drop is large (rapid deterioration signal)." if lang == "en" else "انخفاض MAP الأخير كبير (إشارة تدهور سريع).")
        elif map_drop >= 5:
            reasons.append("MAP is trending down." if lang == "en" else "MAP يتجه للانخفاض.")
    if not (HR != HR):
        if HR > 100:
            reasons.append("HR > 100 bpm (possible compensatory tachycardia)." if lang == "en" else "HR أكبر من 100 (قد يكون تسرّع قلبي تعويضي).")
        elif HR < 50:
            reasons.append("Bradycardia may contribute to instability." if lang == "en" else "بطء القلب قد يساهم في عدم الاستقرار.")
    if not (SpO2 != SpO2):
        if SpO2 < 92:
            reasons.append("SpO2 < 92% (possible hypoxemia contributing to risk)." if lang == "en" else "SpO2 أقل من 92% (قد يساهم نقص الأكسجة في زيادة الخطر).")

    # Model decision
    if alarm:
        reasons.append(
            (f"Model risk_score ({risk:.3f}) ≥ threshold ({threshold:.2f}) → alarm triggered."
             if lang == "en"
             else f"درجة الخطر من النموذج ({risk:.3f}) ≥ العتبة ({threshold:.2f}) → تم إطلاق الإنذار.")
        )
    else:
        reasons.append(
            (f"Model risk_score ({risk:.3f}) < threshold ({threshold:.2f}) → no alarm."
             if lang == "en"
             else f"درجة الخطر من النموذج ({risk:.3f}) < العتبة ({threshold:.2f}) → لا يوجد إنذار.")
        )

    # Recommendations (general, safe)
    if alarm:
        recs.extend([
            "Check cuff/arterial line signal quality and confirm readings.",
            "Assess volume status, bleeding, sepsis, anesthetic depth, vasodilation causes.",
            "Consider clinician-directed interventions per local protocol (fluids/vasopressors).",
            "Increase monitoring frequency and trend review over the next minutes."
        ] if lang == "en" else [
            "تحقق من جودة قياس الضغط (الكفة/الخط الشرياني) وتأكد من القراءات.",
            "قيّم حالة السوائل/النزف/الإنتان/عمق التخدير/توسع الأوعية كأسباب محتملة.",
            "فكر بإجراءات علاجية حسب بروتوكول المستشفى (سوائل/مقبضات أوعية) بقرار الطبيب.",
            "زد تكرار المراقبة وراجع الاتجاهات خلال الدقائق القادمة."
        ])
    else:
        recs.extend([
            "Continue routine monitoring and watch trends.",
            "Reassess if MAP decreases or risk_score increases."
        ] if lang == "en" else [
            "استمر بالمراقبة الروتينية وانتبه للاتجاهات.",
            "أعد التقييم إذا بدأ MAP ينخفض أو ارتفعت درجة الخطر."
        ])

    headline = headline_alarm if alarm else headline_ok

    return {
        "headline": headline,
        "reasons_title": reasons_title,
        "reasons": reasons,
        "rec_title": rec_title,
        "recommendation": recs,
        "disclaimer": disclaimer
    }
