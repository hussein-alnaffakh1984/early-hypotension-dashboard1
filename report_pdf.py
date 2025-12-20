# report_pdf.py
from io import BytesIO
import numpy as np
import pandas as pd

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

# Charts
import matplotlib.pyplot as plt

# Optional Arabic RTL support
def _rtl(text: str) -> str:
    """
    Convert Arabic to RTL-shaped string if arabic_reshaper + bidi are available.
    Falls back to original text if not installed.
    """
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception:
        return text


def _t(lang, en, ar):
    return en if lang == "en" else ar


def _make_chart_png(df_out: pd.DataFrame) -> bytes:
    """
    Create a simple chart (MAP + risk_score) and return PNG bytes.
    """
    buf = BytesIO()
    cols = []
    if "MAP" in df_out.columns:
        cols.append("MAP")
    if "risk_score" in df_out.columns:
        cols.append("risk_score")
    if not cols:
        return b""

    plt.figure()
    x = df_out["time"] if "time" in df_out.columns else np.arange(len(df_out))
    for c in cols:
        plt.plot(x, df_out[c].astype(float), label=c)
    plt.xlabel("time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=140)
    plt.close()
    return buf.getvalue()


def generate_pdf_report(
    df_out: pd.DataFrame,
    patient_info: dict,
    explanation: dict,
    threshold: float,
    drop_text: str,
    lang: str = "en",
) -> bytes:
    """
    Returns PDF bytes.
    """
    pdf_buf = BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=A4)
    w, h = A4

    margin = 1.6 * cm
    y = h - margin

    title = _t(lang, "Hypotension Early Warning Report", "تقرير الإنذار المبكر لهبوط الضغط")
    if lang == "ar":
        title = _rtl(title)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, title)
    y -= 1.0 * cm

    # Patient block
    c.setFont("Helvetica-Bold", 12)
    head = _t(lang, "Patient Information", "معلومات المريض")
    if lang == "ar":
        head = _rtl(head)
    c.drawString(margin, y, head)
    y -= 0.6 * cm

    c.setFont("Helvetica", 11)
    for k in ["Patient ID", "Age", "Sex", "ICU/OR", "Drop Type"]:
        v = patient_info.get(k, "")
        line = f"{k}: {v}"
        if lang == "ar":
            # Show Arabic labels but keep keys English for consistency
            ar_key_map = {
                "Patient ID": "معرف المريض",
                "Age": "العمر",
                "Sex": "الجنس",
                "ICU/OR": "الموقع",
                "Drop Type": "نوع الهبوط"
            }
            line = f"{ar_key_map.get(k,k)}: {v}"
            line = _rtl(line)
        c.drawString(margin, y, line)
        y -= 0.5 * cm

    # Model block
    y -= 0.2 * cm
    c.setFont("Helvetica-Bold", 12)
    head2 = _t(lang, "Model Settings", "إعدادات النموذج")
    if lang == "ar":
        head2 = _rtl(head2)
    c.drawString(margin, y, head2)
    y -= 0.6 * cm
    c.setFont("Helvetica", 11)

    line = _t(lang, f"Threshold: {threshold:.2f}", f"العتبة: {threshold:.2f}")
    if lang == "ar":
        line = _rtl(line)
    c.drawString(margin, y, line)
    y -= 0.5 * cm

    line = _t(lang, f"Drop Type: {drop_text}", f"نوع الهبوط: {drop_text}")
    if lang == "ar":
        line = _rtl(line)
    c.drawString(margin, y, line)
    y -= 0.8 * cm

    # Outcome summary
    latest = df_out.iloc[-1]
    MAP = float(latest["MAP"]) if "MAP" in latest else np.nan
    risk = float(latest["risk_score"]) if "risk_score" in latest else np.nan
    alarm = bool(latest["alarm"]) if "alarm" in latest else False

    c.setFont("Helvetica-Bold", 12)
    head3 = _t(lang, "Current Status", "الحالة الحالية")
    if lang == "ar":
        head3 = _rtl(head3)
    c.drawString(margin, y, head3)
    y -= 0.6 * cm
    c.setFont("Helvetica", 11)

    status_line = _t(
        lang,
        f"MAP: {MAP:.1f} | Risk Score: {risk:.3f} | Alarm: {'YES' if alarm else 'NO'}",
        f"MAP: {MAP:.1f} | درجة الخطر: {risk:.3f} | إنذار: {'نعم' if alarm else 'لا'}"
    )
    if lang == "ar":
        status_line = _rtl(status_line)
    c.drawString(margin, y, status_line)
    y -= 0.9 * cm

    # Explanation
    c.setFont("Helvetica-Bold", 12)
    head4 = _t(lang, "Medical Explanation", "التفسير الطبي")
    if lang == "ar":
        head4 = _rtl(head4)
    c.drawString(margin, y, head4)
    y -= 0.6 * cm

    c.setFont("Helvetica", 11)
    headline = explanation.get("headline", "")
    if lang == "ar":
        headline = _rtl(headline)
    c.drawString(margin, y, headline)
    y -= 0.6 * cm

    reasons_title = explanation.get("reasons_title", _t(lang, "Why?", "لماذا؟"))
    if lang == "ar":
        reasons_title = _rtl(reasons_title)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, reasons_title)
    y -= 0.5 * cm

    c.setFont("Helvetica", 10.5)
    for r in explanation.get("reasons", [])[:10]:
        txt = f"- {r}"
        if lang == "ar":
            txt = _rtl(txt)
        c.drawString(margin, y, txt[:140])  # truncate long
        y -= 0.42 * cm
        if y < 5 * cm:
            c.showPage()
            y = h - margin
            c.setFont("Helvetica", 10.5)

    y -= 0.2 * cm
    rec_title = explanation.get("rec_title", _t(lang, "Recommendation", "التوصيات"))
    if lang == "ar":
        rec_title = _rtl(rec_title)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, rec_title)
    y -= 0.5 * cm

    c.setFont("Helvetica", 10.5)
    for r in explanation.get("recommendation", [])[:10]:
        txt = f"- {r}"
        if lang == "ar":
            txt = _rtl(txt)
        c.drawString(margin, y, txt[:140])
        y -= 0.42 * cm
        if y < 6 * cm:
            c.showPage()
            y = h - margin
            c.setFont("Helvetica", 10.5)

    # Chart
    chart_png = _make_chart_png(df_out)
    if chart_png:
        if y < 10 * cm:
            c.showPage()
            y = h - margin

        c.setFont("Helvetica-Bold", 12)
        head5 = _t(lang, "Trends", "الاتجاهات")
        if lang == "ar":
            head5 = _rtl(head5)
        c.drawString(margin, y, head5)
        y -= 0.6 * cm

        img = ImageReader(BytesIO(chart_png))
        img_w = w - 2 * margin
        img_h = 7.0 * cm
        c.drawImage(img, margin, y - img_h, width=img_w, height=img_h, preserveAspectRatio=True, mask="auto")
        y -= (img_h + 0.6 * cm)

    # Disclaimer
    c.setFont("Helvetica-Oblique", 9.5)
    dis = explanation.get("disclaimer", _t(lang,
                                          "Disclaimer: decision-support only.",
                                          "تنبيه: مساعد قرار فقط."))
    if lang == "ar":
        dis = _rtl(dis)
    c.drawString(margin, 1.3 * cm, dis[:170])

    c.showPage()
    c.save()

    return pdf_buf.getvalue()
