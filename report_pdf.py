# report_pdf.py
import io
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def generate_pdf_report(df_out: pd.DataFrame, patient_info: dict, explanation: dict,
                        threshold: float, drop_text: str, lang: str = "en") -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    y = h - 50

    title = "Hypotension Early Warning Report" if lang == "en" else "تقرير الإنذار المبكر لهبوط الضغط"
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, title)
    y -= 25

    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Threshold: {threshold:.2f}")
    y -= 15
    c.drawString(50, y, f"Drop Type: {drop_text}")
    y -= 20

    # Patient info
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Patient Info" if lang == "en" else "معلومات المريض")
    y -= 15

    c.setFont("Helvetica", 10)
    for k, v in patient_info.items():
        c.drawString(60, y, f"{k}: {v}")
        y -= 13

    y -= 10

    # Summary
    latest = df_out.iloc[-1]
    risk = float(latest.get("risk_score", 0.0))
    alarm = bool(latest.get("alarm", False))

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Model Decision" if lang == "en" else "قرار النموذج")
    y -= 15
    c.setFont("Helvetica", 10)
    c.drawString(60, y, f"Risk Score: {risk:.3f}")
    y -= 13
    c.drawString(60, y, f"Alarm: {'YES' if alarm else 'NO'}")
    y -= 20

    # Explanation
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Explanation" if lang == "en" else "التفسير")
    y -= 15

    c.setFont("Helvetica", 10)
    c.drawString(60, y, explanation.get("headline", ""))
    y -= 15

    c.setFont("Helvetica-Bold", 10)
    c.drawString(60, y, explanation.get("reasons_title", "Why?"))
    y -= 13
    c.setFont("Helvetica", 10)
    for r in explanation.get("reasons", [])[:10]:
        c.drawString(70, y, f"- {r}")
        y -= 12

    y -= 8
    c.setFont("Helvetica-Bold", 10)
    c.drawString(60, y, explanation.get("rec_title", "Recommendation"))
    y -= 13
    c.setFont("Helvetica", 10)
    for r in explanation.get("recommendation", [])[:10]:
        c.drawString(70, y, f"- {r}")
        y -= 12

    y -= 10
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, y, explanation.get("disclaimer", ""))
    y -= 20

    # small table excerpt
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Last Rows (preview)" if lang == "en" else "آخر القيم (معاينة)")
    y -= 15
    c.setFont("Helvetica", 9)

    tail = df_out.tail(5)[["time", "MAP", "HR", "SpO2", "risk_score", "alarm"]].copy()
    lines = tail.to_string(index=False).split("\n")
    for line in lines[:12]:
        c.drawString(50, y, line[:120])
        y -= 11

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()
