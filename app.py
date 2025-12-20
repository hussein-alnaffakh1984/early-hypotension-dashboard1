# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO

import matplotlib.pyplot as plt

from features import extract_features
from gate import apply_gate
from alarm import generate_alarm

from pdf_report import make_pdf_report


# -------------------------------
# Page
# -------------------------------
st.set_page_config(page_title="Hypotension Early Warning Dashboard", layout="wide")
st.title("🫀 Hypotension Early Warning Dashboard")
st.caption("Upload patient CSV → features → (Gate) → model → alarms")


# -------------------------------
# Load artifacts
# -------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    feature_cols = joblib.load("feature_cols.joblib")  # list of column names
    return model, feature_cols

model, FEATURE_COLS = load_artifacts()


# -------------------------------
# Helpers
# -------------------------------
def ensure_required_vitals(df: pd.DataFrame) -> pd.DataFrame:
    # Must have at least these columns; RR optional.
    for col in ["time", "MAP", "HR", "SpO2"]:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")

    if "RR" not in df.columns:
        df["RR"] = np.nan

    # Optional columns for your trained model features (EtCO2)
    if "EtCO2" not in df.columns:
        df["EtCO2"] = np.nan

    return df


def align_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Key fix: keep DataFrame with names, and force same columns & order as training.
    """
    X_aligned = X.reindex(columns=FEATURE_COLS, fill_value=np.nan)
    return X_aligned


def make_medical_explanation(df_raw: pd.DataFrame, df_out: pd.DataFrame, threshold: float, drop_type: str) -> tuple[list[str], list[str]]:
    """
    Returns:
      explanation_lines: why alarm happened
      recommendation_lines: what to do (general, no meds)
    """
    latest = df_out.iloc[-1]
    alarm = bool(latest["alarm"])
    risk = float(latest["risk_score"])

    # trends
    def safe_delta(col: str, k: int):
        if len(df_raw) <= k:
            return np.nan
        return float(df_raw[col].iloc[-1] - df_raw[col].iloc[-1 - k])

    map_now = float(df_raw["MAP"].iloc[-1])
    hr_now  = float(df_raw["HR"].iloc[-1])
    spo2_now = float(df_raw["SpO2"].iloc[-1])
    rr_now = float(df_raw["RR"].iloc[-1]) if "RR" in df_raw.columns else np.nan

    dMAP_1 = safe_delta("MAP", 1)
    dMAP_5 = safe_delta("MAP", min(5, len(df_raw)-1))
    dHR_5  = safe_delta("HR",  min(5, len(df_raw)-1))
    dSpO2_5 = safe_delta("SpO2", min(5, len(df_raw)-1))

    explanation = []
    explanation.append(f"Risk Score وصل إلى {risk:.3f} (العتبة = {threshold:.2f}) ⇒ {'إنذار' if alarm else 'بدون إنذار'}")
    explanation.append(f"MAP الحالي = {map_now:.1f} (Δ1={dMAP_1:.1f}, Δ5={dMAP_5:.1f})")
    explanation.append(f"HR الحالي = {hr_now:.1f} (Δ5={dHR_5:.1f})")
    explanation.append(f"SpO₂ الحالي = {spo2_now:.1f} (Δ5={dSpO2_5:.1f})")
    if not np.isnan(rr_now):
        explanation.append(f"RR الحالي = {rr_now:.1f}")

    # Interpretation rules (lightweight, explainable)
    if map_now < 65:
        explanation.append("MAP أقل من 65 ⇒ هذا يدعم وجود hypotension/قرب حدوثه.")
    if dMAP_5 < -5:
        explanation.append("هبوط سريع في MAP خلال آخر دقائق ⇒ زيادة خطر هبوط قريب.")
    if dHR_5 > 5:
        explanation.append("ارتفاع HR مع هبوط MAP قد يشير لاستجابة تعويضية.")
    if dSpO2_5 < -2:
        explanation.append("انخفاض SpO₂ قد يزيد خطورة الحالة أو يشير لتدهور عام.")

    explanation.append(f"Drop Type المختار: {drop_type} (يؤثر على طريقة عرض التفسير/الحساسية وليس على نموذج ML إلا إذا أُدرج كميزة أثناء التدريب).")

    # Recommendations (general, safe)
    rec = []
    if alarm:
        rec.append("تنبيه الفريق السريري/المسؤول المناوب ومراجعة العلامات الحيوية فورًا.")
        rec.append("تأكيد القراءة (sensor check) وإعادة القياس للتأكد من عدم وجود artifact.")
        rec.append("متابعة MAP trend خلال الدقائق القادمة ومراجعة الأدوية/السوائل حسب بروتوكول القسم.")
        rec.append("اعتبار تقييم سريري شامل (علامات صدمة/نقص حجم/نزف/إنتان) بحسب الحالة.")
    else:
        rec.append("المتابعة المستمرة ومراقبة الاتجاهات (Trends).")
        rec.append("إعادة التقييم إذا بدأ MAP بالهبوط أو ارتفع Risk Score.")

    return explanation, rec


def run_inference(df_raw: pd.DataFrame, threshold: float, use_gate: bool) -> tuple[pd.DataFrame, np.ndarray]:
    df_raw = df_raw.copy()

    # Feature extraction (your existing function)
    X = extract_features(df_raw)  # expected to return DataFrame
    X = align_features(X)

    gate_mask = None
    if use_gate:
        X, gate_mask = apply_gate(X)  # allow either (X) or (X,mask)
        if isinstance(X, tuple):
            # in case apply_gate returns (X,mask)
            X, gate_mask = X

    probs = model.predict_proba(X)[:, 1]
    df_out = df_raw.copy()
    df_out["risk_score"] = probs
    df_out["alarm"] = df_out["risk_score"].apply(lambda x: generate_alarm(x, threshold))

    return df_out, gate_mask


def plot_to_png_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def build_vitals_plot(df: pd.DataFrame) -> bytes:
    fig = plt.figure(figsize=(10, 3.2))
    ax = fig.add_subplot(111)
    for col in ["MAP", "HR", "SpO2"]:
        if col in df.columns:
            ax.plot(df["time"], df[col], label=col)
    if "RR" in df.columns and not df["RR"].isna().all():
        ax.plot(df["time"], df["RR"], label="RR")
    ax.set_title("Raw Vitals")
    ax.set_xlabel("time")
    ax.legend()
    return plot_to_png_bytes(fig)


def build_risk_plot(df_out: pd.DataFrame) -> bytes:
    fig = plt.figure(figsize=(10, 3.2))
    ax = fig.add_subplot(111)
    ax.plot(df_out["time"], df_out["risk_score"], label="risk_score")
    ax.set_title("Alarm Timeline (Risk Score)")
    ax.set_xlabel("time")
    ax.legend()
    return plot_to_png_bytes(fig)


def scenario_threshold(drop_type: str, base_thr: float) -> float:
    """
    For A/B/C comparison we keep SAME model output,
    but show different sensitivity via threshold presets.
    """
    presets = {
        "A: Rapid": max(0.01, base_thr - 0.03),       # more sensitive
        "B: Gradual": base_thr,                       # baseline
        "C: Intermittent": max(0.01, base_thr - 0.01) # slightly more sensitive
    }
    return presets.get(drop_type, base_thr)


# -------------------------------
# Sidebar: Patient + Settings
# -------------------------------
st.sidebar.header("🧾 Patient Summary")
patient_id = st.sidebar.text_input("🧑‍⚕️ Patient ID", value="P-001")
age = st.sidebar.number_input("🎂 Age", min_value=0, max_value=130, value=45, step=1)
sex = st.sidebar.selectbox("⚧ Sex", ["Male", "Female"])
icu_or = st.sidebar.selectbox("🏥 ICU / OR", ["ICU", "OR"])

st.sidebar.header("⚙️ Model Settings")
base_threshold = st.sidebar.slider("Threshold (manual)", 0.01, 0.99, 0.11, 0.01)
use_gate = st.sidebar.checkbox("Enable Gate", value=True)

drop_type = st.sidebar.selectbox("اختيار نوع الهبوط", ["A: Rapid", "B: Gradual", "C: Intermittent"])

st.sidebar.header("🧾 Input Mode")
mode = st.sidebar.radio("Input Mode", ["CSV Upload", "Manual Entry"])

uploaded_file = None
df = None

if mode == "CSV Upload":
    uploaded_file = st.file_uploader("Upload patient CSV file", type=["csv"])
    st.info("CSV must contain at least: time, MAP, HR, SpO2 (RR optional).")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

else:
    st.subheader("✍️ Manual Entry (single row)")
    colA, colB, colC, colD, colE = st.columns(5)
    t = colA.number_input("time", value=0.0, step=1.0)
    MAP = colB.number_input("MAP", value=80.0, step=1.0)
    HR = colC.number_input("HR", value=80.0, step=1.0)
    SpO2 = colD.number_input("SpO2", value=98.0, step=1.0)
    RR = colE.number_input("RR", value=16.0, step=1.0)

    df = pd.DataFrame([{"time": t, "MAP": MAP, "HR": HR, "SpO2": SpO2, "RR": RR}])


# -------------------------------
# Main
# -------------------------------
if df is None:
    st.warning("⬅️ اختر CSV أو Manual Entry للبدء.")
    st.stop()

try:
    df = ensure_required_vitals(df)
except Exception as e:
    st.error(f"Input error: {e}")
    st.stop()

# Charts
st.subheader("📈 Raw Vitals")
st.line_chart(df.set_index("time")[["MAP", "HR", "SpO2"] + (["RR"] if "RR" in df.columns else [])])

# Run inference (base threshold)
try:
    df_out, gate_mask = run_inference(df, threshold=base_threshold, use_gate=use_gate)
except Exception as e:
    st.error("Error during inference:")
    st.exception(e)
    st.stop()

# Show results
st.subheader("🚨 Alarm Timeline")
st.line_chart(df_out.set_index("time")[["risk_score"]])

latest = df_out.iloc[-1]
alarm_now = bool(latest["alarm"])
risk_now = float(latest["risk_score"])

st.subheader("🩺 Current Status")
c1, c2, c3, c4 = st.columns(4)
c1.metric("MAP", f"{float(df['MAP'].iloc[-1]):.1f}")
c2.metric("Risk Score", f"{risk_now:.3f}")
c3.metric("Alarm", "YES 🚨" if alarm_now else "NO ✅")
c4.metric("Drop Type", drop_type.split(":")[0])

# -------------------------------
# 1) Medical Auto Explanation
# -------------------------------
st.subheader("🧠 Medical Explanation (Auto)")
explain_lines, rec_lines = make_medical_explanation(df, df_out, base_threshold, drop_type)

st.markdown("**Why did the model raise the alarm?**")
for line in explain_lines:
    st.write(f"- {line}")

st.markdown("**Recommendation (General):**")
for line in rec_lines:
    st.write(f"- {line}")

st.caption("⚠️ هذا نظام دعم قرار فقط وليس تشخيصًا طبيًا.")

# -------------------------------
# 3) A/B/C Comparison (same data)
# -------------------------------
st.subheader("🧪 A / B / C Scenario Comparison (same data)")
types = ["A: Rapid", "B: Gradual", "C: Intermittent"]
rows = []

for ttype in types:
    thr = scenario_threshold(ttype, base_threshold)
    # same model outputs, different threshold => different alarm sensitivity
    alarm_any = (df_out["risk_score"] >= thr).any()
    alarm_first_time = df_out.loc[df_out["risk_score"] >= thr, "time"].iloc[0] if alarm_any else None
    rows.append({
        "Scenario": ttype,
        "Threshold Used": round(thr, 3),
        "Any Alarm?": "YES" if alarm_any else "NO",
        "First Alarm Time": alarm_first_time
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True)

# -------------------------------
# 2) PDF Report
# -------------------------------
st.subheader("📄 Generate PDF Report")

vitals_png = build_vitals_plot(df)
risk_png = build_risk_plot(df_out)

patient_info = {
    "patient_id": patient_id,
    "age": age,
    "sex": sex,
    "icu_or": icu_or,
    "drop_type": drop_type,
    "threshold": base_threshold,
    "use_gate": use_gate,
}

summary = {
    "MAP": float(df["MAP"].iloc[-1]),
    "HR": float(df["HR"].iloc[-1]),
    "SpO2": float(df["SpO2"].iloc[-1]),
    "RR": (float(df["RR"].iloc[-1]) if "RR" in df.columns and not pd.isna(df["RR"].iloc[-1]) else "NA"),
    "risk_score": round(risk_now, 3),
    "alarm": "YES" if alarm_now else "NO",
}

pdf_bytes = make_pdf_report(
    patient_info=patient_info,
    summary=summary,
    vitals_img_bytes=vitals_png,
    risk_img_bytes=risk_png,
    recommendation_lines=rec_lines,
)

st.download_button(
    "⬇️ Download PDF Report",
    data=pdf_bytes,
    file_name=f"hypotension_report_{patient_id}.pdf",
    mime="application/pdf",
)
