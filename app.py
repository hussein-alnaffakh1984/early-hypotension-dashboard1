import streamlit as st
import pandas as pd
import numpy as np
import joblib

from features import extract_features
from gate import apply_gate
from alarm import generate_alarm

st.set_page_config(page_title="Hypotension Early Warning System", layout="wide")
st.title("🫀 Hypotension Early Warning Dashboard")
st.caption("Upload a patient CSV → features → (Gate) → model → alarms")

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    feature_cols = joblib.load("feature_cols.joblib")
    return model, feature_cols

model, feature_cols = load_artifacts()

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Alarm threshold", 0.01, 0.99, 0.15)
use_gate = st.sidebar.checkbox("Enable Gate", value=True)

uploaded_file = st.file_uploader("Upload patient CSV file", type=["csv"])

# ---------- Main ----------
if uploaded_file is None:
    st.info("⬅️ Upload a patient CSV file to start")
    st.stop()

df = pd.read_csv(uploaded_file)

# Basic checks
missing = [c for c in ["MAP", "HR"] if c not in df.columns]
if missing:
    st.error(f"Your CSV is missing required columns: {missing}")
    st.stop()

st.subheader("📈 Raw Vitals")
cols_to_plot = [c for c in ["MAP", "HR", "SpO2"] if c in df.columns]
st.line_chart(df[cols_to_plot])

# Features
X = extract_features(df)

# Gate
if use_gate:
    X = apply_gate(X)

# Ensure same columns as training
for c in feature_cols:
    if c not in X.columns:
        X[c] = 0.0
X = X[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

# Predict
probs = model.predict_proba(X)[:, 1]
df["risk_score"] = probs

# Alarm
df["alarm"] = df["risk_score"].apply(lambda x: generate_alarm(x, threshold))

st.subheader("🚨 Risk Score Timeline")
st.line_chart(df[["risk_score"]])

latest = df.iloc[-1]
st.subheader("🩺 Current Status")
c1, c2, c3 = st.columns(3)
c1.metric("MAP", f"{latest['MAP']:.1f}")
c2.metric("Risk Score", f"{latest['risk_score']:.2f}")
c3.metric("Alarm", "YES 🚨" if latest["alarm"] else "NO ✅")

st.subheader("📄 Download Results")
out = df.copy()
st.download_button(
    "Download CSV with risk_score + alarm",
    out.to_csv(index=False).encode("utf-8"),
    file_name="patient_with_alarms.csv",
    mime="text/csv",
)
