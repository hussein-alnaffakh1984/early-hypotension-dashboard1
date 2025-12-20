# pdf_report.py
from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import cm


def make_pdf_report(
    patient_info: dict,
    summary: dict,
    vitals_img_bytes: bytes,
    risk_img_bytes: bytes,
    recommendation_lines: list[str],
) -> bytes:
    """
    Returns a PDF as bytes.
    vitals_img_bytes / risk_img_bytes should be PNG bytes.
    """
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, h - 2 * cm, "Hypotension Early Warning Report")

    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, h - 2.7 * cm, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    # Patient Info
    y = h - 3.6 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Patient Information")
    y -= 0.6 * cm

    c.setFont("Helvetica", 10)
    lines = [
        f"Patient ID: {patient_info.get('patient_id', '')}",
        f"Age: {patient_info.get('age', '')}",
        f"Sex: {patient_info.get('sex', '')}",
        f"ICU/OR: {patient_info.get('icu_or', '')}",
        f"Drop Type (selected): {patient_info.get('drop_type', '')}",
        f"Threshold: {patient_info.get('threshold', '')}",
        f"Gate: {patient_info.get('use_gate', '')}",
    ]
    for line in lines:
        c.drawString(2 * cm, y, line)
        y -= 0.45 * cm

    # Summary
    y -= 0.2 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Model Decision Summary")
    y -= 0.6 * cm

    c.setFont("Helvetica", 10)
    sum_lines = [
        f"Latest MAP: {summary.get('MAP', 'NA')}",
        f"Latest HR: {summary.get('HR', 'NA')}",
        f"Latest SpO2: {summary.get('SpO2', 'NA')}",
        f"Latest RR: {summary.get('RR', 'NA')}",
        f"Risk Score: {summary.get('risk_score', 'NA')}",
        f"Alarm: {summary.get('alarm', 'NA')}",
    ]
    for line in sum_lines:
        c.drawString(2 * cm, y, line)
        y -= 0.45 * cm

    # Charts
    y -= 0.2 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Charts")
    y -= 0.7 * cm

    # Vitals image
    if vitals_img_bytes:
        img = ImageReader(BytesIO(vitals_img_bytes))
        img_w = w - 4 * cm
        img_h = 7.0 * cm
        c.drawImage(img, 2 * cm, y - img_h, width=img_w, height=img_h, preserveAspectRatio=True, anchor='sw')
        y -= (img_h + 0.6 * cm)

    # Risk image
    if risk_img_bytes:
        img = ImageReader(BytesIO(risk_img_bytes))
        img_w = w - 4 * cm
        img_h = 7.0 * cm
        c.drawImage(img, 2 * cm, y - img_h, width=img_w, height=img_h, preserveAspectRatio=True, anchor='sw')
        y -= (img_h + 0.6 * cm)

    # Recommendations
    if y < 6 * cm:
        c.showPage()
        y = h - 2.5 * cm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Recommendation")
    y -= 0.6 * cm

    c.setFont("Helvetica", 10)
    for r in recommendation_lines:
        c.drawString(2 * cm, y, f"- {r}")
        y -= 0.45 * cm

    # Disclaimer
    y -= 0.4 * cm
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(
        2 * cm, y,
        "Disclaimer: This report is for decision support only and not a medical diagnosis."
    )

    c.showPage()
    c.save()

    buf.seek(0)
    return buf.getvalue()
