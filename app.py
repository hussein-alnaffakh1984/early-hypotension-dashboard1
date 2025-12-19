import streamlit as st
import pandas as pd
import numpy as np
import joblib

from features import extract_features
from gate import apply_gate
from alarm import generate_alarm, classify_drop_type

# ===============================
# Page config
# ===============================
st.set_page_config(page_title="Hypotension Early Warning System", layout="wide")
st.title("🫀 Hypotension Early Warning Dashboard")
st.caption("Upload CSV → features → (Gate) → model → alarms")

# ===============================
# Load model + feature columns
# ===============================
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    feature_cols = joblib.load("feature_cols.joblib")
    return model, feature_cols

model, feature_cols = load_artifacts()

# ===============================
# Sidebar: Patient info + settings
# ===============================
st.sidebar.header("🧑‍⚕️ Patient Info")
patient_id = st.sidebar.text_input("Patient ID", value="P-001")
age = st.sidebar.number_input("🎂 Age", min_value=0, max_value=120, value=45, step=1)
sex = st.sidebar.selectbox("⚧ Sex", ["Male", "Female", "Other/Unknown"], index=0)
location = st.sidebar.selectbox("🏥 ICU / OR", ["ICU", "OR", "Ward/Other"], index=0)

st.sidebar.divider()
st.sidebar.header("⚙️ Model Settings")
threshold = st.sidebar.slider("Threshold (Alarm)", 0.01, 0.99, 0.15, 0.01)
use_gate = st.sidebar.checkbox("Enable Gate", value=True)

drop_choice = st.sidebar.selectbox(
    "Choose hypotension type (for reporting)",
    ["Auto (from signal)", "A: Rapid", "B: Gradual", "C: Intermittent"],
    index=0
)

# ===============================
# Input mode
# ===============================
st.sidebar.divider()
mode = st.sidebar.radio("Input Mode", ["Upload CSV", "Manual Input"], index=0)

# Session state for manual rows
if "manual_df" not in st.session_state:
    st.session_state.manual_df = pd.DataFrame(columns=["time", "MAP", "HR", "SpO2", "RR"])

# ===============================
# Manual Input UI
# ===============================
def manual_input_panel():
    st.subheader("✍️ Manual Input (No CSV)")
    c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1, 1])

    with c1:
        t = st.number_input("time", value=float(len(st.session_state.manual_df)), step=1.0)
    with c2:
        mapv = st.number_input("MAP", value=80.0, step=1.0)
    with c3:
        hrv = st.number_input("HR", value=80.0, step=1.0)
    with c4:
        spo2v = st.number_input("SpO2", value=98.0, step=1.0)
    with c5:
        rrv = st.number_input("RR", value=16.0, step=1.0)

    b1, b2, b3 = st.columns([1, 1, 2])
    with b1:
        add_row = st.button("➕ Add row")
    with b2:
        clear_rows = st.button("🧹 Clear")

    if add_row:
        new_row = {"time": t, "MAP": mapv, "HR": hrv, "SpO2": spo2v, "RR": rrv}
        st.session_state.manual_df = pd.concat(
            [st.session_state.manual_df, pd.DataFrame([new_row])],
            ignore_index=True
        ).sort_values("time").reset_index(drop=True)

    if clear_rows:
        st.session_state.manual_df = pd.DataFrame(columns=["time", "MAP", "HR", "SpO2", "RR"])

    st.write("Current manual table:")
    st.dataframe(st.session_state.manual_df, use_container_width=True)

    return st.session_state.manual_df.copy()

# ===============================
# CSV Upload UI
# ===============================
def upload_panel():
    st.subheader("📤 Upload patient CSV")
    up = st.file_uploader("CSV columns should include: time, MAP, HR, SpO2, (optional RR)", type=["csv"])
    if up is None:
        st.info("⬅️ Upload a CSV to start, or switch to Manual Input.")
        return None
    df = pd.read_csv(up)
    return df

# ===============================
# Get input dataframe
# ===============================
if mode == "Upload CSV":
    df_raw = upload_panel()
else:
    df_raw = manual_input_panel()

if df_raw is None or len(df_raw) == 0:
    st.stop()

# ===============================
# Normalize columns
# ===============================
df = df_raw.copy()
for col in ["time", "MAP", "HR", "SpO2", "RR"]:
    if col not in df.columns:
        df[col] = np.nan

# Ensure numeric
for col in ["time", "MAP", "HR", "SpO2", "RR"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Basic check
needed = ["time", "MAP", "HR", "SpO2"]
missing = [c for c in needed if df[c].isna().all()]
if missing:
    st.error(f"Your data is missing values for columns: {missing}. Please provide them.")
    st.stop()

# ===============================
# Display raw vitals
# ===============================
st.subheader("📈 Raw Vitals")
st.line_chart(df.set_index("time")[["MAP", "HR", "SpO2"]], use_container_width=True)

# ===============================
# Feature extraction → align columns → (Gate) → predict
# ===============================
X = extract_features(df)

# Align to training feature columns (fix mismatch)
X = X.reindex(columns=feature_cols, fill_value=0)

if use_gate:
    Xg = apply_gate(X)
    # apply_gate can return filtered rows
    if isinstance(Xg, pd.DataFrame) and len(Xg) > 0:
        X = Xg.reindex(columns=feature_cols, fill_value=0)

probs = model.predict_proba(X)[:, 1]

# Make output timeline match length of probs
out = df.iloc[-len(probs):].copy().reset_index(drop=True)
out["risk_score"] = probs
out["alarm"] = out["risk_score"].apply(lambda p: generate_alarm(p, threshold))

# ===============================
# Hypotension type (A/B/C)
# ===============================
auto_type = classify_drop_type(out["MAP"].values, out["time"].values)
chosen_type = auto_type if drop_choice == "Auto (from signal)" else drop_choice.split(":")[0].strip()

# ===============================
# Results
# ===============================
st.subheader("🚨 Risk Score & Alarm")
st.line_chart(out.set_index("time")[["risk_score"]], use_container_width=True)

latest = out.iloc[-1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("MAP", f"{latest['MAP']:.1f}")
c2.metric("Risk Score", f"{latest['risk_score']:.2f}")
c3.metric("Alarm", "YES 🚨" if bool(latest["alarm"]) else "NO ✅")
c4.metric("Drop Type", chosen_type)

st.subheader("🧾 Patient Report (preview)")
report = {
    "Patient ID": patient_id,
    "Age": age,
    "Sex": sex,
    "Location": location,
    "Threshold": threshold,
    "Gate": "Enabled" if use_gate else "Disabled",
    "Drop Type (selected)": chosen_type,
    "Drop Type (auto)": auto_type,
    "Last MAP": float(latest["MAP"]),
    "Last Risk Score": float(latest["risk_score"]),
    "Alarm": bool(latest["alarm"]),
}
st.json(report)

st.subheader("📋 Last 15 rows")
st.dataframe(out.tail(15), use_container_width=True)
