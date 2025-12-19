import streamlit as st
import pandas as pd
import numpy as np
import joblib

from features import extract_features
from gate import apply_gate
from alarm import generate_alarm

st.set_page_config(page_title="Hypotension Early Warning Dashboard", layout="wide")

MODEL_PATH = "model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🫀 Hypotension Early Warning Dashboard")
st.caption("Upload patient CSV → features → (Gate) → model → alarms")

# =========================
# Sidebar: Patient Info
# =========================
st.sidebar.header("🧾 Patient Summary")

patient_id = st.sidebar.text_input("🧑‍⚕️ Patient ID", value="P-001")
age = st.sidebar.number_input("🎂 Age", min_value=0, max_value=120, value=45, step=1)
sex = st.sidebar.selectbox("⚧ Sex", ["Male", "Female"])
unit = st.sidebar.selectbox("🏥 ICU / OR", ["ICU", "OR"])

drop_type = st.sidebar.selectbox(
    "اختيار نوع الهبوط",
    ["A: Rapid", "B: Gradual", "C: Intermittent"],
    index=0
)

st.sidebar.header("⚙️ إعدادات النموذج")
threshold = st.sidebar.slider("Threshold يدوي", 0.01, 0.99, 0.15, 0.01)
use_gate = st.sidebar.checkbox("Enable Gate", value=True)

st.sidebar.header("🧾 طريقة الإدخال")
input_mode = st.sidebar.radio("Input Mode", ["CSV Upload", "Manual Entry"], index=0)

# =========================
# Input
# =========================
df = None

if input_mode == "CSV Upload":
    uploaded_file = st.file_uploader("Upload patient CSV file", type=["csv"])
    st.info("CSV must contain at least: time, MAP, HR, SpO2 (RR optional). EtCO2 optional.")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

else:
    st.subheader("✍️ إدخال يدوي (بدون CSV)")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        t = st.number_input("time", value=0.0, step=1.0)
    with col2:
        MAP = st.number_input("MAP", value=80.0, step=1.0)
    with col3:
        HR = st.number_input("HR", value=78.0, step=1.0)
    with col4:
        SpO2 = st.number_input("SpO2", value=98.0, step=1.0)
    with col5:
        RR = st.number_input("RR", value=16.0, step=1.0)

    # نبني DataFrame بسيط من صف واحد (ممكن لاحقاً تضيف زر "Add Row")
    df = pd.DataFrame([{"time": t, "MAP": MAP, "HR": HR, "SpO2": SpO2, "RR": RR}])

# =========================
# Inference
# =========================
def run_inference(df_in: pd.DataFrame, threshold: float, use_gate: bool):
    df_in = df_in.copy()

    required = ["time", "MAP", "HR", "SpO2"]
    missing = [c for c in required if c not in df_in.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # عرض القياسات الخام
    show_cols = [c for c in ["MAP", "HR", "SpO2", "RR"] if c in df_in.columns]
    st.subheader("📈 Raw Vitals")
    st.line_chart(df_in[show_cols])

    # استخراج Features (يعطي 26 عمود مطابق للـ feature_cols.joblib)
    X = extract_features(df_in)   # DataFrame بأسماء أعمدة
    # IMPORTANT: لا تحول إلى numpy حتى لا يصير feature_names mismatch

    # Gate
    gate_mask = None
    if use_gate:
        # gate يحتاج بعض أعمدة من الفيتشر + MAP الحالية
        tmp = df_in.copy()
        # نضيف MAP_drop_2m للـ gate لو متوفر
        if "MAP_drop_2m" in X.columns:
            tmp["MAP_drop_2m"] = X["MAP_drop_2m"]
        gate_mask = apply_gate(tmp)
    else:
        gate_mask = pd.Series([True] * len(df_in), index=df_in.index)

    # Prediction
    probs = model.predict_proba(X)[:, 1]

    # طبق gate: إذا False نخلي risk = 0
    probs = np.where(gate_mask.to_numpy(), probs, 0.0)

    df_out = df_in.copy()
    df_out["risk_score"] = probs
    df_out["alarm"] = df_out["risk_score"].apply(lambda r: generate_alarm(r, threshold))

    return df_out, gate_mask

if df is None:
    st.stop()

try:
    df_out, gate_mask = run_inference(df, threshold=threshold, use_gate=use_gate)

    st.subheader("🚨 Alarm Timeline")
    st.line_chart(df_out[["risk_score"]])

    st.subheader("🩺 Current Status")
    latest = df_out.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAP", f"{float(latest['MAP']):.1f}")
    c2.metric("Risk Score", f"{float(latest['risk_score']):.3f}")
    c3.metric("Alarm", "YES 🚨" if bool(latest["alarm"]) else "NO ✅")
    c4.metric("Drop Type", drop_type.split(":")[0].strip())

    st.subheader("🧾 Patient Info")
    st.write({
        "Patient ID": patient_id,
        "Age": int(age),
        "Sex": sex,
        "ICU/OR": unit,
        "Drop Type": drop_type
    })

    with st.expander("Show output table"):
        st.dataframe(df_out)

except Exception as e:
    st.error("Error during inference:")
    st.exception(e)
