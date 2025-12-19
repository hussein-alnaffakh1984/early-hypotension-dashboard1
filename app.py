import streamlit as st
import pandas as pd
import numpy as np
import joblib

from features import extract_features_timeseries
from gate import apply_gate
from alarm import refractory_alarm

st.set_page_config(page_title="Hypotension Early Warning Dashboard", layout="wide")

st.title("🫀 Hypotension Early Warning Dashboard")
st.caption("Upload patient CSV → features → (Gate) → model → alarms")

# -----------------------------
# Load model + feature cols
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")  # RF model (بدون Pipeline)
    feat_cols = joblib.load("feature_cols.joblib")  # list of 26 feature names
    return model, feat_cols

def safe_float_cols(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X

# -----------------------------
# Sidebar - patient info + settings
# -----------------------------
st.sidebar.header("🧾 Patient Summary")
patient_id = st.sidebar.text_input("🧑‍⚕️ Patient ID", value="P-001")
age = st.sidebar.number_input("🎂 Age", min_value=0, max_value=120, value=45)
sex = st.sidebar.selectbox("⚧ Sex", ["Male", "Female"])
unit = st.sidebar.selectbox("🏥 ICU / OR", ["ICU", "OR"])

st.sidebar.header("⚙️ Model Settings")
threshold = st.sidebar.slider("Threshold (Alarm)", 0.01, 0.99, 0.15, 0.01)
refract_sec = st.sidebar.slider("Refractory (sec)", 0, 600, 180, 30)

drop_type = st.sidebar.selectbox(
    "اختيار نوع الهبوط",
    ["A: Rapid", "B: Gradual", "C: Intermittent"],
    index=0
)

use_gate = st.sidebar.checkbox("Enable Gate", value=True)
gate_drop_thr = st.sidebar.slider("Gate: MAP_drop_2m threshold", -30.0, 0.0, -5.0, 0.5)
gate_map_thr  = st.sidebar.slider("Gate: MAP_m60 threshold", 40.0, 120.0, 75.0, 1.0)

st.sidebar.header("📥 Input Mode")
mode = st.sidebar.radio("Choose input", ["Upload CSV", "Manual input"], index=0)

# -----------------------------
# Input
# -----------------------------
df = None

if mode == "Upload CSV":
    uploaded = st.file_uploader("Upload patient CSV file", type=["csv"])
    st.info("CSV must contain at least: time, MAP, HR, SpO2 (RR optional).")
    if uploaded is not None:
        df = pd.read_csv(uploaded)

else:
    st.subheader("✍️ Manual Input (no CSV)")
    c1, c2, c3, c4 = st.columns(4)
    MAP = c1.number_input("MAP", value=80.0)
    HR  = c2.number_input("HR", value=78.0)
    SpO2 = c3.number_input("SpO2", value=98.0)
    RR  = c4.number_input("RR", value=16.0)

    # نصنع سلسلة 2 دقائق حتى نقدر نحسب rolling features
    n = 180
    df = pd.DataFrame({
        "time": np.arange(n, dtype=float),
        "MAP": np.full(n, MAP, dtype=float),
        "HR": np.full(n, HR, dtype=float),
        "SpO2": np.full(n, SpO2, dtype=float),
        "RR": np.full(n, RR, dtype=float),
    })

# -----------------------------
# Inference pipeline
# -----------------------------
def run_inference(df_in: pd.DataFrame):
    model, feat_cols = load_artifacts()

    # ensure minimal cols
    needed = {"time", "MAP", "HR", "SpO2"}
    if not needed.issubset(set(df_in.columns)):
        raise ValueError(f"Missing required columns: {sorted(list(needed - set(df_in.columns)))}")

    df_in = df_in.copy()
    df_in["time"] = pd.to_numeric(df_in["time"], errors="coerce")
    df_in = df_in.sort_values("time").reset_index(drop=True)

    # features
    X = extract_features_timeseries(df_in)
    X = safe_float_cols(X)

    # reorder columns exactly as training
    missing = [c for c in feat_cols if c not in X.columns]
    extra = [c for c in X.columns if c not in feat_cols]
    if missing:
        raise ValueError(f"Feature mismatch: missing columns in X: {missing[:10]} ...")
    X = X[feat_cols]

    # gate mask
    if use_gate:
        gate_mask = apply_gate(df_in, X, drop_thr=gate_drop_thr, map_thr=gate_map_thr)
    else:
        gate_mask = np.ones(len(X), dtype=bool)

    # model predict (IMPORTANT: pass numpy float)
    X_np = X.to_numpy(dtype=np.float32)
    probs = model.predict_proba(X_np)[:, 1].astype(float)

    # alarms with refractory
    time_sec = df_in["time"].to_numpy(dtype=float)
    alarms = refractory_alarm(probs, thr=float(threshold), refract_sec=float(refract_sec), time_sec=time_sec)

    out = df_in.copy()
    out["risk_score"] = probs
    out["gate_pass"] = gate_mask.astype(int)
    out["alarm"] = alarms.astype(int)

    return out

# -----------------------------
# UI Output
# -----------------------------
if df is None:
    st.stop()

try:
    df_out = run_inference(df)

    st.subheader("📈 Raw Vitals")
    show_cols = [c for c in ["MAP", "HR", "SpO2", "RR"] if c in df_out.columns]
    st.line_chart(df_out.set_index("time")[show_cols])

    st.subheader("🧠 Risk score & Alarm")
    st.line_chart(df_out.set_index("time")[["risk_score"]])

    latest = df_out.iloc[-1]
    st.subheader("🩺 Current Status")
    a, b, c, d = st.columns(4)
    a.metric("Patient", patient_id)
    b.metric("MAP", f"{float(latest['MAP']):.1f}")
    c.metric("Risk", f"{float(latest['risk_score']):.3f}")
    d.metric("Alarm", "YES 🚨" if int(latest["alarm"]) == 1 else "NO ✅")

    st.write("**Metadata**:", {"Age": int(age), "Sex": sex, "Unit": unit, "DropType": drop_type})

    st.subheader("📋 Output table (download)")
    st.dataframe(df_out.tail(50), use_container_width=True)

    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download alarms CSV", data=csv_bytes, file_name=f"{patient_id}_alarms.csv", mime="text/csv")

except Exception as e:
    st.error(f"Error: {e}")
    st.stop()
