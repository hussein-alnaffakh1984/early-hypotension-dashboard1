import streamlit as st
import pandas as pd
import numpy as np
import pickle

from features import extract_features
from gate import apply_gate
from alarm import generate_alarm

# ===============================
# Page config
# ===============================
st.set_page_config(
    page_title="Hypotension Early Warning System",
    layout="wide"
)

st.title("🫀 Hypotension Early Warning Dashboard")

st.markdown("""
This dashboard provides **early warning for hypotension**
using vital signs and a trained ML model.
""")

# ===============================
# Load model
# ===============================
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ===============================
# Sidebar
# ===============================
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Alarm threshold", 0.05, 0.9, 0.15)
use_gate = st.sidebar.checkbox("Enable Gate", value=True)

uploaded_file = st.file_uploader(
    "Upload patient CSV file",
    type=["csv"]
)

# ===============================
# Main logic
# ===============================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📈 Raw Vitals")
    st.line_chart(df[["MAP", "HR", "SpO2"]])

    # Feature extraction
    X = extract_features(df)

    # Gate
    if use_gate:
        X = apply_gate(X)

    # Prediction
    probs = model.predict_proba(X)[:, 1]
    df["risk_score"] = probs

    # Alarm
    df["alarm"] = df["risk_score"].apply(
        lambda x: generate_alarm(x, threshold)
    )

    st.subheader("🚨 Alarm Timeline")
    st.line_chart(df[["risk_score"]])

    # Summary
    latest = df.iloc[-1]

    st.subheader("🩺 Current Status")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAP", f"{latest['MAP']:.1f}")
    col2.metric("Risk Score", f"{latest['risk_score']:.2f}")
    col3.metric("Alarm", "YES 🚨" if latest["alarm"] else "NO ✅")

else:
    st.info("⬅️ Upload a patient CSV file to start")
