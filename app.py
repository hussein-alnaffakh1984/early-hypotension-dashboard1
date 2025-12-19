import streamlit as st
import pandas as pd
import numpy as np
import joblib

from features import extract_features
from gate import apply_gate
from alarm import generate_alarm


# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="Hypotension Early Warning Dashboard", layout="wide")
st.title("🫀 Hypotension Early Warning Dashboard")
st.caption("Upload patient CSV → features → (Gate) → model → alarms")


# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    feature_cols = joblib.load("feature_cols.joblib")  # list of expected feature names
    return model, feature_cols

model, feature_cols = load_artifacts()


# -----------------------------
# Sidebar: Patient info + settings
# -----------------------------
st.sidebar.header("🧾 Patient Summary")

patient_id = st.sidebar.text_input("🧑‍⚕️ Patient ID", value="P-001")
age = st.sidebar.number_input("🎂 Age", min_value=0, max_value=120, value=45, step=1)
sex = st.sidebar.selectbox("⚧ Sex", ["Male", "Female"])
unit = st.sidebar.selectbox("🏥 ICU / OR", ["ICU", "OR"])
drop_type = st.sidebar.selectbox("اختيار نوع الهبوط", ["A: Rapid", "B: Gradual", "C: Intermittent"])

st.sidebar.header("⚙️ إعدادات النموذج")
threshold = st.sidebar.slider("Threshold يدوي", 0.01, 0.99, 0.15, 0.01)
use_gate = st.sidebar.checkbox("Enable Gate", value=True)

st.sidebar.header("🧾 طريقة الإدخال")
mode = st.sidebar.radio("Input Mode", ["CSV Upload", "Manual Entry"], index=0)


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure df has at least time, MAP, HR, SpO2; RR optional."""
    df = df.copy()
    if "time" not in df.columns:
        df["time"] = np.arange(len(df), dtype=float)

    for c in ["MAP", "HR", "SpO2"]:
        if c not in df.columns:
            df[c] = np.nan

    if "RR" not in df.columns:
        df["RR"] = np.nan

    # optional EtCO2 (feature generator can handle missing)
    if "EtCO2" not in df.columns:
        df["EtCO2"] = np.nan

    return df


def align_features(X: pd.DataFrame, expected_cols: list) -> pd.DataFrame:
    """Force X to have exactly expected_cols in same order."""
    X = X.copy()
    for c in expected_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[expected_cols]
    return X


def auto_drop_type(df: pd.DataFrame) -> str:
    """Simple heuristic for A/B/C based on MAP trend."""
    s = pd.to_numeric(df["MAP"], errors="coerce").astype(float)
    if len(s) < 5:
        return "Unknown"
    drop_2m = (s.rolling(10, min_periods=1).max() - s).iloc[-1]  # assuming ~2s step -> 10 ~20s (light heuristic)
    recent_slope = (s.iloc[-1] - s.iloc[-5])  # last 5 points
    recent_std = s.iloc[-10:].std()

    if drop_2m >= 15 or recent_slope <= -10:
        return "A (Rapid)"
    if recent_std is not None and recent_std >= 6:
        return "C (Intermittent)"
    return "B (Gradual)"


def run_inference(df_raw: pd.DataFrame):
    df_raw = ensure_columns(df_raw)

    # show raw chart
    st.subheader("📈 Raw Vitals")
    show_cols = [c for c in ["MAP", "HR", "SpO2", "RR"] if c in df_raw.columns]
    st.line_chart(df_raw[show_cols])

    # features
    X = extract_features(df_raw)
    X = align_features(X, feature_cols)

    gate_mask = None
    if use_gate:
        X, gate_mask = apply_gate(X)

    # IMPORTANT: pass DataFrame with correct feature names
    probs = model.predict_proba(X)[:, 1]
    df_out = df_raw.copy()
    df_out["risk_score"] = probs
    df_out["alarm"] = df_out["risk_score"].apply(lambda x: generate_alarm(x, threshold))

    return df_out, gate_mask


# -----------------------------
# Main: CSV or Manual
# -----------------------------
if mode == "CSV Upload":
    st.subheader("Upload patient CSV file")
    uploaded_file = st.file_uploader("Upload patient CSV file", type=["csv"])

    st.info("CSV must contain at least: time, MAP, HR, SpO2 (RR optional).", icon="ℹ️")

    if uploaded_file is None:
        st.stop()

    df = pd.read_csv(uploaded_file)
    df_out, gate_mask = run_inference(df)

else:
    st.subheader("إدخال يدوي (بدون CSV)")

    colA, colB, colC, colD = st.columns(4)
    with colA:
        t = st.number_input("time", value=0.0, step=1.0)
        map_v = st.number_input("MAP", value=80.0, step=1.0)
    with colB:
        hr_v = st.number_input("HR", value=80.0, step=1.0)
        spo2_v = st.number_input("SpO2", value=98.0, step=1.0)
    with colC:
        rr_v = st.number_input("RR", value=16.0, step=1.0)
        etco2_v = st.number_input("EtCO2", value=35.0, step=1.0)
    with colD:
        points = st.number_input("How many points?", min_value=5, max_value=300, value=30, step=5)
        dt = st.number_input("dt (seconds)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)

    # Create a simple synthetic series ending at current values
    times = np.arange(points) * dt
    df = pd.DataFrame({
        "time": times,
        "MAP": np.linspace(map_v + 5, map_v, points),
        "HR": np.linspace(hr_v - 5, hr_v, points),
        "SpO2": np.linspace(spo2_v, spo2_v, points),
        "RR": np.linspace(rr_v, rr_v, points),
        "EtCO2": np.linspace(etco2_v, etco2_v, points),
    })

    df_out, gate_mask = run_inference(df)


# -----------------------------
# Outputs
# -----------------------------
st.subheader("🚨 Alarm Timeline")
st.line_chart(df_out[["risk_score"]])

latest = df_out.iloc[-1]
auto_type = auto_drop_type(df_out)

st.subheader("🩺 Current Status")
c1, c2, c3, c4 = st.columns(4)
c1.metric("MAP", f"{latest['MAP']:.1f}")
c2.metric("Risk Score", f"{latest['risk_score']:.3f}")
c3.metric("Alarm", "YES 🚨" if bool(latest["alarm"]) else "NO ✅")
c4.metric("Auto Drop Type", auto_type)

with st.expander("🔎 Patient Info"):
    st.write(
        {
            "Patient ID": patient_id,
            "Age": int(age),
            "Sex": sex,
            "ICU/OR": unit,
            "Selected Drop Type": drop_type,
            "Threshold": float(threshold),
            "Gate Enabled": bool(use_gate),
        }
    )

if gate_mask is not None:
    st.subheader("🧱 Gate Summary")
    st.write(f"Rows kept by gate: {int(gate_mask.sum())} / {len(gate_mask)}")

st.subheader("📄 Output Table (last 30 rows)")
st.dataframe(df_out.tail(30), use_container_width=True)
