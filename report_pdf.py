# report_pdf.py
from __future__ import annotations
import io
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


def _plot_to_image(df: pd.DataFrame, cols, title: str):
    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_subplot(111)
    for c in cols:
        if c in df.columns:
            ax.plot(df["time"], df[c], label=c)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_pdf_report(
    df_out: pd.DataFrame,
    patient_info: dict,
    explanation: dict,
    threshold: float,
    drop_text: str,
) -> bytes:
    """
    Returns PDF bytes. Use in Streamlit with st.download_button.
    df_out expected columns: time, MAP, HR, SpO2, (RR optional), risk_score, alarm
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, h-2*cm, "Hypotension Early Warning Report")

    # Patient info
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, h-3.2*cm, "Patient Information")
    c.setFont("Helvetica", 11)

    y = h-4.0*cm
    for k in ["Patient ID", "Age", "Sex", "ICU/OR"]:
        c.drawString(2*cm, y, f"{k}: {patient_info.get(k,'')}")
        y -= 0.6*cm

    c.drawString(2*cm, y, f"Drop Type: {drop_text}")
    y -= 0.6*cm
    c.drawString(2*cm, y, f"Threshold: {threshold:.2f}")
    y -= 0.8*cm

    # Model decision
    last = df_out.iloc[-1]
    alarm = bool(last.get("alarm", False))
    risk = float(last.get("risk_score", 0.0))

    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "Model Decision")
    y -= 0.7*cm
    c.setFont("Helvetica", 11)
    c.drawString(2*cm, y, f"Final Risk Score: {risk:.3f}")
    y -= 0.6*cm
    c.drawString(2*cm, y, f"Alarm: {'YES' if alarm else 'NO'}")
    y -= 0.9*cm

    # Explanation
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "Automatic Medical Explanation")
    y -= 0.7*cm
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, y, explanation.get("headline", ""))
    y -= 0.6*cm

    c.setFont("Helvetica-Bold", 10)
    c.drawString(2*cm, y, "Key Reasons:")
    y -= 0.5*cm
    c.setFont("Helvetica", 10)
    for r in explanation.get("reasons", [])[:7]:
        c.drawString(2.2*cm, y, f"- {r[:110]}")
        y -= 0.45*cm

    y -= 0.3*cm
    c.setFont("Helvetica-Bold", 10)
    c.drawString(2*cm, y, "Recommendation:")
    y -= 0.5*cm
    c.setFont("Helvetica", 10)
    for r in explanation.get("recommendation", [])[:6]:
        c.drawString(2.2*cm, y, f"- {r[:110]}")
        y -= 0.45*cm

    y -= 0.4*cm

    # New page for charts if needed
    c.showPage()

    # Charts
    vitals_img = _plot_to_image(df_out, ["MAP", "HR", "SpO2", "RR"], "Raw Vitals")
    risk_img = _plot_to_image(df_out, ["risk_score"], "Risk Score Timeline")

    c.setFont("Helvetica-Bold", 13)
    c.drawString(2*cm, h-2*cm, "Charts")

    # Place images
    img1 = ImageReader(vitals_img)
    img2 = ImageReader(risk_img)

    c.drawImage(img1, 2*cm, h-10*cm, width=w-4*cm, height=7*cm, preserveAspectRatio=True, anchor='nw')
    c.drawImage(img2, 2*cm, h-19*cm, width=w-4*cm, height=7*cm, preserveAspectRatio=True, anchor='nw')

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(2*cm, 1.8*cm, explanation.get("disclaimer", ""))

    c.save()
    pdf = buf.getvalue()
    buf.close()
    return pdf
