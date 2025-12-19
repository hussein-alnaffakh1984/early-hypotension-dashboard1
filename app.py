import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# Page config
# ===============================
st.set_page_config(
    page_title="Hypotension Early Warning Dashboard",
    layout="wide"
)

# ===============================
# Safe imports (if files exist)
# ===============================
def _fallback_extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback feature extractor (simple).
    IMPORTANT: We will align output columns to feature_cols.joblib anyway.
    """
    df = df.copy()
    # Ensure required vitals exist
    for col in ["MAP", "HR", "SpO2", "RR"]:
        if col not in df.columns:
            df[col] = np.nan

    # Basic rolling stats (very simple)
    w = 5
    feats = pd.DataFrame(index=df.index)
    feats["MAP"] = df["MAP"]
    feats["HR"] = df["HR"]
    feats["SpO2"] = df["SpO2"]
    feats["RR"] = df["RR"]

    feats["MAP_m5"] = df["MAP"].rolling(w, min_periods=1).mean()
    feats["MAP_s5"] = df["MAP"].rolling(w, min_periods=1).std().fillna(0)
    feats["HR_m5"] = df["HR"].rolling(w, min_periods=1).mean()
    feats["SpO2_m5"] = df["SpO2"].rolling(w, min_periods=1).mean()
    feats["RR_m5"] = df["RR"].rolling(w, min_periods=1).mean()

    feats["MAP_d1"] = df["MAP"].diff().fillna(0)
    feats["HR_d1"] = df["HR"].diff().fillna(0)
    feats["SpO2_d1"] = df["SpO2"].diff().fillna(0)
    feats["RR_d1"] = df["RR"].diff().fillna(0)

    return feats


def _fallback_apply_gate(X: pd.DataFrame, df_raw: pd.DataFrame) -> pd.Series:
    """
    Fallback gate: only allow prediction when MAP is dropping or low.
    Returns boolean mask.
    """
    if "MAP" not in df_raw.columns:
        return pd.Series([True] * len(X), index=X.index)
    mapv = pd.to_numeric(df_raw["MAP"], errors="coerce")
    drop = mapv.diff().fillna(0)
    mask = (mapv < 75) | (drop < -0.5)
    return mask.fillna(False)


def _fallback_generate_alarm(prob: float, thr: float) -> bool:
    return bool(prob >= thr)


# Try importing your real modules if present
try:
    from features import extract_features  # expected: extract_features(df)->DataFrame
except Exception:
    extract_features = _fallback_extract_features

try:
    from gate import apply_gate  # expected: apply_gate(X, df_raw)->mask or X
except Exception:
    apply_gate = None

try:
    from alarm import generate_alarm  # expected: generate_alarm(prob, thr)->bool
except Exception:
    generate_alarm = _fallback_generate_alarm


# ===============================
# Model + feature columns
# ===============================
@st.cache_resource
def load_model_and_cols():
    model = joblib.load("model.joblib")
    feature_cols = joblib.load("feature_cols.joblib")  # MUST be list of names
    return model, feature_cols

model, FEATURE_COLS = load_model_and_cols()


def align_features(X: pd.DataFrame, feature_cols) -> pd.DataFrame:
    """
    Make X match training columns exactly (names + order).
    Missing cols -> add NaN (imputer inside pipeline will handle)
    Extra cols -> drop
    """
    X = X.copy()
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[feature_cols]
    return X


def run_inference(df_raw: pd.DataFrame, threshold: float, use_gate: bool):
    """
    Returns (df_out, mask_used_for_gate)
    """
    # 1) Extract features
    X = extract_features(df_raw)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # 2) Align to model expected features
    X = align_features(X, FEATURE_COLS)

    # 3) Gate
    if use_gate:
        if apply_gate is not None:
            try:
                gate_mask = apply_gate(X, df_raw)  # can return mask OR X filtered
                if isinstance(gate_mask, (pd.Series, np.ndarray, list)):
                    gate_mask = pd.Series(gate_mask, index=X.index).astype(bool)
                else:
                    # if apply_gate returns X itself
                    gate_mask = pd.Series([True] * len(X), index=X.index)
            except Exception:
                gate_mask = _fallback_apply_gate(X, df_raw)
        else:
            gate_mask = _fallback_apply_gate(X, df_raw)
    else:
        gate_mask = pd.Series([True] * len(X), index=X.index)

    # 4) Predict (IMPORTANT: pass DataFrame, not numpy)
    probs = model.predict_proba(X)[:, 1]

    df_out = df_raw.copy()
    df_out["risk_score"] = probs
    df_out["gate"] = gate_mask.values

    # 5) Alarm (apply threshold + gate)
    def _alarm_row(p, g):
        if use_gate and (not bool(g)):
            return False
        return generate_alarm(float(p), float(threshold))

    df_out["alarm"] = [
        _alarm_row(p, g) for p, g in zip(df_out["risk_score"], df_out["gate"])
    ]

    return df_out, gate_mask


# ===============================
# UI Header
# ===============================
st.title("🫀 Hypotension Early Warning Dashboard")
st.caption("Upload patient CSV → features → (Gate) → model → alarms")

# ===============================
# Sidebar: Patient info + settings
# ===============================
st.sidebar.header("🧑‍⚕️ Patient Info")
patient_id = st.sidebar.text_input("Patient ID", value="P-001")
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=45, step=1)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
location = st.sidebar.selectbox("ICU / OR", ["ICU", "OR", "Ward", "Other"])

st.sidebar.header("⚙️ Model Settings")
threshold = st.sidebar.slider("Threshold", 0.01, 0.99, 0.15, 0.01)
use_gate = st.sidebar.checkbox("Enable Gate", value=True)

st.sidebar.header("📉 Drop Type (Research)")
drop_type = st.sidebar.radio(
    "Select hypotension drop type",
    ["A: Rapid", "B: Gradual", "C: Intermittent"],
    index=0
)

st.sidebar.markdown("---")
mode = st.sidebar.radio("Input Mode", ["Upload CSV", "Manual Input"], index=0)

# ===============================
# Helpers
# ===============================
def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    # Required
    for c in ["time", "MAP", "HR", "SpO2", "RR"]:
        if c not in df.columns:
            df[c] = np.nan
    # Ensure numeric
    for c in ["time", "MAP", "HR", "SpO2", "RR"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)
    return df

# ===============================
# Main: Upload CSV
# ===============================
if mode == "Upload CSV":
    uploaded = st.file_uploader("Upload patient CSV file", type=["csv"])

    st.info("CSV must contain at least: time, MAP, HR, SpO2 (RR optional).")

    if uploaded is None:
        st.stop()

    df = pd.read_csv(uploaded)
    df = ensure_columns(df)

# ===============================
# Main: Manual Input
# ===============================
else:
    st.subheader("✍️ Manual Input (No CSV)")
    st.caption("Add points one by one, then run the model.")

    if "manual_rows" not in st.session_state:
        st.session_state.manual_rows = []

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        t = st.number_input("time", value=float(len(st.session_state.manual_rows)), step=1.0)
    with c2:
        mapv = st.number_input("MAP", value=80.0, step=1.0)
    with c3:
        hrv = st.number_input("HR", value=80.0, step=1.0)
    with c4:
        spo2v = st.number_input("SpO2", value=98.0, step=1.0)
    with c5:
        rrv = st.number_input("RR", value=16.0, step=1.0)

    b1, b2 = st.columns(2)
    with b1:
        if st.button("➕ Add point"):
            st.session_state.manual_rows.append(
                {"time": t, "MAP": mapv, "HR": hrv, "SpO2": spo2v, "RR": rrv}
            )
    with b2:
        if st.button("🧹 Clear"):
            st.session_state.manual_rows = []

    df = pd.DataFrame(st.session_state.manual_rows)
    if df.empty:
        st.warning("Add at least 5 points to see meaningful output.")
        st.stop()

    df = ensure_columns(df)

# ===============================
# Display patient summary
# ===============================
st.markdown("### 🧾 Patient Summary")
p1, p2, p3, p4, p5 = st.columns(5)
p1.metric("Patient ID", patient_id)
p2.metric("Age", f"{age}")
p3.metric("Sex", sex)
p4.metric("ICU/OR", location)
p5.metric("Drop Type", drop_type.split(":")[0])

# ===============================
# Run inference
# ===============================
df_out, gate_mask = run_inference(df, threshold=threshold, use_gate=use_gate)

# ===============================
# Charts
# ===============================
st.subheader("📈 Raw Vitals")
show_cols = ["MAP", "HR", "SpO2"]
if "RR" in df_out.columns:
    show_cols.append("RR")
st.line_chart(df_out[show_cols])

st.subheader("🧠 Risk Score & Alarm")
st.line_chart(df_out[["risk_score"]])

# Alarm timeline (as 0/1)
alarm_series = df_out["alarm"].astype(int)
st.line_chart(pd.DataFrame({"alarm": alarm_series}))

# ===============================
# Current status
# ===============================
st.subheader("🩺 Current Status")
latest = df_out.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("MAP", f"{latest['MAP']:.1f}" if pd.notna(latest["MAP"]) else "NA")
c2.metric("Risk Score", f"{latest['risk_score']:.3f}")
c3.metric("Gate", "ON ✅" if bool(latest["gate"]) else "OFF ⛔")
c4.metric("Alarm", "YES 🚨" if bool(latest["alarm"]) else "NO ✅")

# ===============================
# Debug (optional)
# ===============================
with st.expander("🔍 Debug (Feature Compatibility)"):
    st.write("Model expects #features:", len(FEATURE_COLS))
    X_dbg = extract_features(df_out)
    if not isinstance(X_dbg, pd.DataFrame):
        X_dbg = pd.DataFrame(X_dbg)
    st.write("Extracted feature shape:", X_dbg.shape)
    missing = [c for c in FEATURE_COLS if c not in X_dbg.columns]
    st.write("Missing features count:", len(missing))
    st.write("First 30 missing:", missing[:30])

# ===============================
# Export results
# ===============================
st.subheader("⬇️ Export")
export_df = df_out.copy()
export_df["patient_id"] = patient_id
export_df["age"] = age
export_df["sex"] = sex
export_df["location"] = location
export_df["drop_type"] = drop_type

csv_bytes = export_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download results CSV",
    data=csv_bytes,
    file_name=f"{patient_id}_hypotension_results.csv",
    mime="text/csv"
)
