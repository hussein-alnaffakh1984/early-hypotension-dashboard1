import streamlit as st
import pandas as pd
import numpy as np
import joblib

from features import build_feature_matrix, get_expected_feature_columns
from gate import apply_gate
from alarm import generate_alarm


# ===============================
# Page config
# ===============================
st.set_page_config(page_title="Hypotension Early Warning Dashboard", layout="wide")
st.title("🫀 Hypotension Early Warning Dashboard")
st.caption("Upload patient CSV → features → (Gate) → model → alarms")

# ===============================
# Load model + expected cols
# ===============================
@st.cache_resource
def load_model():
    # model file name in repo
    return joblib.load("model.joblib")

model = load_model()

expected_cols = get_expected_feature_columns(model)




# ===============================
# Sidebar: Patient Info
# ===============================
st.sidebar.header("🧾 Patient Summary")

patient_id = st.sidebar.text_input("🧑‍⚕️ Patient ID", value="P-001")
age = st.sidebar.number_input("🎂 Age", min_value=0, max_value=130, value=45, step=1)
sex = st.sidebar.selectbox("⚧ Sex", ["Male", "Female"])
location = st.sidebar.selectbox("🏥 ICU / OR", ["ICU", "OR"])

st.sidebar.divider()

# ===============================
# Sidebar: Model Settings
# ===============================
st.sidebar.header("⚙️ Model Settings")
threshold = st.sidebar.slider("Threshold (manual)", 0.01, 0.99, 0.11)
use_gate = st.sidebar.checkbox("Enable Gate", value=True)

drop_type = st.sidebar.selectbox(
    "اختيار نوع الهبوط",
    ["A: Rapid", "B: Gradual", "C: Intermittent"],
    index=0
)
drop_key = drop_type.split(":")[0].strip()  # "A" / "B" / "C"

st.sidebar.divider()

# ===============================
# Sidebar: Input Mode
# ===============================
st.sidebar.header("طريقة الإدخال")
input_mode = st.sidebar.radio("Input Mode", ["CSV Upload", "Manual Entry"], index=0)

# ===============================
# Helpers
# ===============================
def normalize_input_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure required columns exist.
    Required: time, MAP, HR, SpO2
    Optional: RR, EtCO2
    """
    df = df.copy()

    # normalize column names
    df.columns = [c.strip() for c in df.columns]

    # required
    required = ["time", "MAP", "HR", "SpO2"]
    missing_req = [c for c in required if c not in df.columns]
    if missing_req:
        raise ValueError(f"CSV is missing required columns: {missing_req}")

    # optional
    if "RR" not in df.columns:
        df["RR"] = np.nan
    if "EtCO2" not in df.columns:
        df["EtCO2"] = np.nan

    # force numeric
    for c in ["time", "MAP", "HR", "SpO2", "RR", "EtCO2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # sort by time
    df = df.sort_values("time").reset_index(drop=True)
    return df


def run_inference(df_raw: pd.DataFrame, threshold: float, use_gate: bool, drop_key: str):
    """
    Returns:
      df_out : df with risk_score + alarm
      gate_mask : boolean mask aligned to feature rows (may be None)
      X : features dataframe used for model
    """
    df = normalize_input_df(df_raw)

    # Feature extraction (always returns DataFrame with expected_cols)
    X = build_feature_matrix(df, expected_cols=expected_cols)

    gate_mask = None
    if use_gate:
        # ✅ IMPORTANT: apply_gate may return either X or (X, mask). We handle BOTH safely.
        gate_result = apply_gate(X, drop_key=drop_key)

        if isinstance(gate_result, tuple):
            # could be (X)?? no, tuple implies multiple
            if len(gate_result) >= 2:
                X, gate_mask = gate_result[0], gate_result[1]
            else:
                X = gate_result[0]
                gate_mask = None
        else:
            X = gate_result
            gate_mask = None

    # predict
    probs = model.predict_proba(X)[:, 1]

    df_out = df.copy()
    # align length: features are row-wise (same length as df)
    df_out["risk_score"] = probs
    df_out["alarm"] = df_out["risk_score"].apply(lambda s: generate_alarm(s, threshold))

    return df_out, gate_mask, X


def compare_drop_types(df_raw: pd.DataFrame, threshold: float, use_gate: bool):
    """
    Run same data with A/B/C and return a comparison dataframe.
    """
    rows = []
    for key, label in [("A", "A: Rapid"), ("B", "B: Gradual"), ("C", "C: Intermittent")]:
        try:
            df_out, _, _ = run_inference(df_raw, threshold=threshold, use_gate=use_gate, drop_key=key)
            last = df_out.iloc[-1]
            rows.append({
                "Drop Type": label,
                "Last MAP": float(last["MAP"]),
                "Last Risk": float(last["risk_score"]),
                "Alarm": "YES 🚨" if bool(last["alarm"]) else "NO ✅"
            })
        except Exception as e:
            rows.append({
                "Drop Type": label,
                "Last MAP": np.nan,
                "Last Risk": np.nan,
                "Alarm": f"ERROR: {e}"
            })
    return pd.DataFrame(rows)


# ===============================
# Main UI
# ===============================
df_input = None

if input_mode == "CSV Upload":
    uploaded_file = st.file_uploader("Upload patient CSV file", type=["csv"])
    st.info("CSV must contain at least: time, MAP, HR, SpO2 (RR optional).")

    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)

else:
    st.subheader("🧾 Manual Entry (بدون CSV)")
    st.caption("أدخل قيَم الحيويات (سطر واحد أو أكثر). إذا تريد سلسلة زمنية، زيد عدد النقاط.")

    n_points = st.number_input("عدد النقاط الزمنية", min_value=1, max_value=300, value=16, step=1)

    colA, colB = st.columns(2)
    with colA:
        start_time = st.number_input("Start time", value=0.0)
        step_time = st.number_input("Time step", value=1.0)
    with colB:
        map_start = st.number_input("MAP start", value=82.0)
        map_end = st.number_input("MAP end", value=56.0)

    hr_start = st.number_input("HR start", value=78.0)
    hr_end = st.number_input("HR end", value=110.0)
    spo2_start = st.number_input("SpO2 start", value=98.0)
    spo2_end = st.number_input("SpO2 end", value=91.0)

    rr_start = st.number_input("RR start (optional)", value=16.0)
    rr_end = st.number_input("RR end (optional)", value=28.0)

    if st.button("Generate Manual Timeseries"):
        t = np.arange(n_points, dtype=float) * float(step_time) + float(start_time)
        df_input = pd.DataFrame({
            "time": t,
            "MAP": np.linspace(map_start, map_end, n_points),
            "HR": np.linspace(hr_start, hr_end, n_points),
            "SpO2": np.linspace(spo2_start, spo2_end, n_points),
            "RR": np.linspace(rr_start, rr_end, n_points),
        })


# ===============================
# Run + Display
# ===============================
if df_input is None:
    st.info("⬅️ اختر طريقة إدخال ثم وفّر بيانات.")
    st.stop()

try:
    df_norm = normalize_input_df(df_input)

    st.subheader("📈 Raw Vitals")
    chart_cols = ["MAP", "HR", "SpO2"]
    if "RR" in df_norm.columns:
        chart_cols.append("RR")
    st.line_chart(df_norm[chart_cols])

    df_out, gate_mask, X = run_inference(df_norm, threshold=threshold, use_gate=use_gate, drop_key=drop_key)

    st.subheader("🚨 Alarm Timeline")
    st.line_chart(df_out[["risk_score"]])

    latest = df_out.iloc[-1]
    st.subheader("🩺 Current Status")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAP", f"{latest['MAP']:.1f}")
    c2.metric("Risk Score", f"{latest['risk_score']:.3f}")
    c3.metric("Alarm", "YES 🚨" if latest["alarm"] else "NO ✅")
    c4.metric("Drop Type", drop_key)

    # تفسير طبي بسيط
    st.subheader("🧠 Automated Medical Explanation (basic)")
    explanation = []
    if latest["MAP"] < 65:
        explanation.append("MAP is below 65 mmHg (hypotension threshold).")
    if latest["HR"] > 100:
        explanation.append("HR is elevated (possible compensatory tachycardia).")
    if latest["risk_score"] >= threshold:
        explanation.append(f"Model risk_score ≥ threshold ({threshold:.2f}), so alarm triggered.")
    if not explanation:
        explanation.append("Vitals are within acceptable range and risk_score below threshold.")

    st.write("• " + "\n• ".join(explanation))

    # Show expected model columns
    with st.expander("Show expected model columns"):
        st.write(list(expected_cols))

    with st.expander("Show extracted feature matrix (head)"):
        st.dataframe(X.head(10))

    # مقارنة A/B/C
    st.subheader("🔁 Compare A / B / C (same data)")
    comp_df = compare_drop_types(df_norm, threshold=threshold, use_gate=use_gate)
    st.dataframe(comp_df, use_container_width=True)

    # تحميل نتائج كـ CSV
    st.download_button(
        "⬇️ Download output CSV (with risk/alarm)",
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name=f"{patient_id}_output.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error("Error during inference:")
    st.exception(e)
