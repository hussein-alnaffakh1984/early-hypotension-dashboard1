import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from features import extract_features_from_timeseries, extract_features_from_single_row
from gate import apply_gate
from alarm import alarm_from_score, classify_drop_type

# ===============================
# Page config
# ===============================
st.set_page_config(page_title="Hypotension Early Warning Dashboard", layout="wide")
st.title("🫀 Hypotension Early Warning Dashboard")
st.caption("Upload CSV → features → (Gate) → model → alarms")

# ===============================
# Load model + feature cols
# ===============================
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")               # ✅ use joblib (NOT pickle)
    feature_cols = joblib.load("feature_cols.joblib") # list of feature names
    return model, list(feature_cols)

model, FEATURE_COLS = load_artifacts()

def align_features(X: pd.DataFrame) -> pd.DataFrame:
    """Fix feature mismatch: drop extras, reindex to training cols, numeric, no NaNs."""
    # drop common non-features if exist
    for c in ["time", "case_id", "hypo", "hypo_event", "future_hypo", "event_onsets", "gate"]:
        if c in X.columns:
            X = X.drop(columns=[c])

    # reindex strictly to training feature set
    X = X.reindex(columns=FEATURE_COLS, fill_value=0)

    # force numeric and clean
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X.astype("float64")
    return X

# ===============================
# Sidebar: patient info + settings
# ===============================
st.sidebar.header("🧑‍⚕️ Patient Info")
patient_id = st.sidebar.text_input("Patient ID", value="P-001")
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=45, step=1)
sex = st.sidebar.selectbox("Sex", ["M", "F"])
unit = st.sidebar.selectbox("ICU / OR", ["ICU", "OR"])

st.sidebar.header("⚙️ Model Settings")
threshold = st.sidebar.slider("Threshold", 0.01, 0.99, 0.15, 0.01)
use_gate = st.sidebar.checkbox("Enable Gate", value=True)

drop_type = st.sidebar.selectbox(
    "Drop Type",
    ["A: Rapid", "B: Gradual", "C: Intermittent"],
    index=0
)

mode = st.sidebar.radio("Input Mode", ["Upload CSV", "Manual Input"], index=0)

st.sidebar.divider()
st.sidebar.write("✅ Model loaded")
st.sidebar.write(f"Features expected: **{len(FEATURE_COLS)}**")

# ===============================
# Helpers
# ===============================
def run_inference_on_timeseries(df: pd.DataFrame):
    # basic checks
    required_cols = ["MAP", "HR", "SpO2"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"CSV missing columns: {missing}. Required at least: {required_cols} (RR optional).")
        return

    # show vitals
    show_cols = [c for c in ["MAP", "HR", "SpO2", "RR"] if c in df.columns]
    st.subheader("📈 Raw Vitals")
    st.line_chart(df[show_cols])

    # feature extraction
    X = extract_features_from_timeseries(df)

    # align to training features (SOLVES mismatch)
    X = align_features(X)

    # gate
    if use_gate:
        X = apply_gate(X)
        X = align_features(X)  # keep alignment after gate

    # predict (use numpy to avoid strict name checking in some pipelines)
    probs = model.predict_proba(X.to_numpy())[:, 1]

    out = df.copy()
    out["risk_score"] = probs
    out["alarm"] = out["risk_score"].apply(lambda s: alarm_from_score(s, threshold))
    out["drop_pred"] = classify_drop_type(out, map_col="MAP")

    # dashboard
    st.subheader("🚨 Risk Score Timeline")
    st.line_chart(out[["risk_score"]])

    # current status
    latest = out.iloc[-1]
    st.subheader("🩺 Current Status")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAP", f"{float(latest['MAP']):.1f}")
    c2.metric("Risk Score", f"{float(latest['risk_score']):.3f}")
    c3.metric("Alarm", "YES 🚨" if bool(latest["alarm"]) else "NO ✅")
    c4.metric("Drop Type (auto)", str(latest["drop_pred"]))

    # patient summary block
    st.subheader("📄 Patient Summary")
    st.write(
        {
            "Patient ID": patient_id,
            "Age": age,
            "Sex": sex,
            "ICU/OR": unit,
            "Chosen Drop Type (manual)": drop_type,
            "Threshold": threshold,
            "Gate Enabled": use_gate,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    # download report
    st.subheader("⬇️ Download")
    report = out[["risk_score", "alarm"] + show_cols].copy()
    report.insert(0, "patient_id", patient_id)
    report.insert(1, "age", age)
    report.insert(2, "sex", sex)
    report.insert(3, "unit", unit)
    report.insert(4, "manual_drop_type", drop_type)

    csv_bytes = report.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download report CSV",
        data=csv_bytes,
        file_name=f"{patient_id}_report.csv",
        mime="text/csv",
    )

def run_inference_manual(map_v, hr_v, spo2_v, rr_v):
    row = {"MAP": map_v, "HR": hr_v, "SpO2": spo2_v, "RR": rr_v}
    X = extract_features_from_single_row(row)
    X = align_features(X)

    if use_gate:
        X = apply_gate(X)
        X = align_features(X)

    prob = float(model.predict_proba(X.to_numpy())[:, 1][0])
    alarm_flag = alarm_from_score(prob, threshold)

    st.subheader("🩺 Manual Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAP", f"{map_v:.1f}")
    c2.metric("Risk Score", f"{prob:.3f}")
    c3.metric("Alarm", "YES 🚨" if alarm_flag else "NO ✅")

    st.subheader("📄 Patient Summary")
    st.write(
        {
            "Patient ID": patient_id,
            "Age": age,
            "Sex": sex,
            "ICU/OR": unit,
            "Chosen Drop Type (manual)": drop_type,
            "Threshold": threshold,
            "Gate Enabled": use_gate,
            "Manual inputs": row,
        }
    )

# ===============================
# UI: Input
# ===============================
if mode == "Upload CSV":
    st.subheader("Upload patient CSV file")
    uploaded = st.file_uploader("CSV must include MAP, HR, SpO2 (RR optional). time optional.", type=["csv"])
    if uploaded is None:
        st.info("⬅️ Upload a CSV to start.")
    else:
        df = pd.read_csv(uploaded)
        run_inference_on_timeseries(df)

else:
    st.subheader("Manual Input (single reading)")
    colA, colB, colC, colD = st.columns(4)
    with colA:
        map_v = st.number_input("MAP", min_value=0.0, max_value=250.0, value=75.0, step=1.0)
    with colB:
        hr_v = st.number_input("HR", min_value=0.0, max_value=250.0, value=80.0, step=1.0)
    with colC:
        spo2_v = st.number_input("SpO2", min_value=0.0, max_value=100.0, value=98.0, step=1.0)
    with colD:
        rr_v = st.number_input("RR", min_value=0.0, max_value=60.0, value=16.0, step=1.0)

    run_inference_manual(map_v, hr_v, spo2_v, rr_v)
