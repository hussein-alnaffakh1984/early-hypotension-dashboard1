import streamlit as st
import pandas as pd
import numpy as np
import joblib

from features import extract_features
from gate import apply_gate
from alarm import generate_alarm

# ======================================
# Page config
# ======================================
st.set_page_config(
    page_title="Hypotension Early Warning Dashboard",
    layout="wide"
)

st.title("🫀 Hypotension Early Warning Dashboard")
st.caption("Upload patient CSV → features → (Gate) → model → alarms")

# ======================================
# Load model
# ======================================
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

model = load_model()

# Try to get expected feature count
EXPECTED_N_FEATURES = getattr(model, "n_features_in_", None)

# ======================================
# Sidebar – Patient Info
# ======================================
st.sidebar.header("🧑‍⚕️ Patient Information")

patient_id = st.sidebar.text_input("Patient ID", "P-001")
age = st.sidebar.number_input("Age", 0, 120, 45)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
location = st.sidebar.selectbox("ICU / OR", ["ICU", "OR"])

drop_type = st.sidebar.selectbox(
    "Drop Type",
    ["A: Rapid", "B: Gradual", "C: Intermittent"]
)

# ======================================
# Sidebar – Model Settings
# ======================================
st.sidebar.header("⚙️ Model Settings")

threshold = st.sidebar.slider("Alarm Threshold", 0.01, 0.9, 0.15)
use_gate = st.sidebar.checkbox("Enable Gate", value=True)

# ======================================
# Data Input
# ======================================
st.subheader("📥 Data Input")

input_mode = st.radio(
    "Choose input mode:",
    ["Upload CSV", "Manual Entry"]
)

# ======================================
# Manual Input
# ======================================
if input_mode == "Manual Entry":
    col1, col2, col3, col4 = st.columns(4)

    MAP = col1.number_input("MAP", 30.0, 120.0, 75.0)
    HR = col2.number_input("HR", 30.0, 200.0, 85.0)
    SpO2 = col3.number_input("SpO2", 50.0, 100.0, 96.0)
    RR = col4.number_input("RR (optional)", 5.0, 40.0, 18.0)

    df = pd.DataFrame({
        "time": [0],
        "MAP": [MAP],
        "HR": [HR],
        "SpO2": [SpO2],
        "RR": [RR],
    })

# ======================================
# CSV Upload
# ======================================
else:
    uploaded_file = st.file_uploader(
        "Upload patient CSV file",
        type=["csv"]
    )

    if uploaded_file is None:
        st.info("⬅️ Upload a patient CSV file to start")
        st.stop()

    df = pd.read_csv(uploaded_file)

# ======================================
# Validate columns
# ======================================
required_cols = {"time", "MAP", "HR", "SpO2"}
if not required_cols.issubset(df.columns):
    st.error(f"CSV must contain at least: {required_cols}")
    st.stop()

# ======================================
# Raw vitals
# ======================================
st.subheader("📈 Raw Vitals")
plot_cols = ["MAP", "HR", "SpO2"]
st.line_chart(df[plot_cols])

# ======================================
# Feature extraction
# ======================================
X = extract_features(df)

# Apply Gate
if use_gate:
    X = apply_gate(X)

# ======================================
# 🔴 CRITICAL FIX: force NumPy & correct shape
# ======================================
if EXPECTED_N_FEATURES is not None:
    if X.shape[1] < EXPECTED_N_FEATURES:
        for i in range(EXPECTED_N_FEATURES - X.shape[1]):
            X[f"_pad_{i}"] = 0.0
    elif X.shape[1] > EXPECTED_N_FEATURES:
        X = X.iloc[:, :EXPECTED_N_FEATURES]

X_np = X.to_numpy(dtype=float)

# ======================================
# Prediction
# ======================================
probs = model.predict_proba(X_np)[:, 1]

df_out = df.copy()
df_out["risk_score"] = probs
df_out["alarm"] = df_out["risk_score"].apply(
    lambda x: generate_alarm(x, threshold)
)

# ======================================
# Results
# ======================================
st.subheader("🚨 Alarm Timeline")
st.line_chart(df_out["risk_score"])

latest = df_out.iloc[-1]

st.subheader("🩺 Current Status")

c1, c2, c3 = st.columns(3)
c1.metric("MAP", f"{latest['MAP']:.1f}")
c2.metric("Risk Score", f"{latest['risk_score']:.2f}")
c3.metric(
    "Alarm",
    "YES 🚨" if latest["alarm"] else "NO ✅"
)

# ======================================
# Patient Summary
# ======================================
st.subheader("🧾 Patient Summary")

st.write({
    "Patient ID": patient_id,
    "Age": age,
    "Sex": sex,
    "Location": location,
    "Drop Type": drop_type,
    "Threshold": threshold,
    "Gate Enabled": use_gate
})
