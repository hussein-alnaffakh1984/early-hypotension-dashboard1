import streamlit as st
import pandas as pd
import joblib

from features import extract_features

# ======================
# Page config
# ======================
st.set_page_config(
    page_title="Hypotension Early Warning Dashboard",
    layout="wide"
)

st.title("🫀 Hypotension Early Warning Dashboard")
st.caption("Upload patient CSV → features → model → alarms")

# ======================
# Load model & columns
# ======================
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    feature_cols = joblib.load("feature_cols.joblib")
    return model, feature_cols

model, FEATURE_COLS = load_artifacts()

# ======================
# Sidebar – Patient info
# ======================
st.sidebar.header("🧾 Patient Summary")

patient_id = st.sidebar.text_input("Patient ID", "P-001")
age = st.sidebar.number_input("Age", 1, 120, 45)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
unit = st.sidebar.selectbox("ICU / OR", ["ICU", "OR"])

drop_type = st.sidebar.selectbox(
    "Drop Type",
    ["A – Rapid", "B – Gradual", "C – Intermittent"]
)

threshold = st.sidebar.slider(
    "Alarm Threshold", 0.05, 0.9, 0.15
)

# ======================
# Upload CSV
# ======================
uploaded_file = st.file_uploader(
    "Upload patient CSV file",
    type=["csv"]
)

st.info("CSV must contain: time, MAP, HR, SpO2 (RR optional)")

if uploaded_file is None:
    st.stop()

# ======================
# Load data
# ======================
df = pd.read_csv(uploaded_file)

required = {"MAP", "HR", "SpO2"}
if not required.issubset(df.columns):
    st.error(f"Missing required columns: {required}")
    st.stop()

# ======================
# Raw vitals
# ======================
st.subheader("📈 Raw Vitals")
st.line_chart(df[["MAP", "HR", "SpO2"]])

# ======================
# Feature extraction
# ======================
X = extract_features(df)

# 🔴 ALIGN FEATURES EXACTLY
X = X.reindex(columns=FEATURE_COLS)

# ======================
# Prediction
# ======================
probs = model.predict_proba(X)[:, 1]
df["risk_score"] = probs
df["alarm"] = df["risk_score"] >= threshold

# ======================
# Outputs
# ======================
st.subheader("🚨 Risk Timeline")
st.line_chart(df["risk_score"])

latest = df.iloc[-1]

st.subheader("🩺 Current Status")
c1, c2, c3 = st.columns(3)

c1.metric("MAP", f"{latest['MAP']:.1f}")
c2.metric("Risk Score", f"{latest['risk_score']:.2f}")
c3.metric("Alarm", "YES 🚨" if latest["alarm"] else "NO ✅")
