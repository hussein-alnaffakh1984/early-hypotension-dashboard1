import streamlit as st
import pandas as pd
import numpy as np
import joblib

from gate import apply_gate
from alarm import generate_alarm

st.set_page_config(page_title="Hypotension Early Warning Dashboard", layout="wide")

MODEL_PATH = "model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

def get_expected_feature_names(m):
    # أغلب موديلات sklearn بعد التدريب تخزن أسماء الأعمدة هنا
    if hasattr(m, "feature_names_in_"):
        return list(m.feature_names_in_)
    # fallback: حاول من آخر خطوة في الـ pipeline
    try:
        if hasattr(m, "named_steps"):
            last = list(m.named_steps.values())[-1]
            if hasattr(last, "feature_names_in_"):
                return list(last.feature_names_in_)
    except Exception:
        pass
    return []

def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def build_X_like_training(df_in: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    """
    يبني DataFrame بنفس أعمدة التدريب وبنفس الترتيب.
    يعالج الأعمدة الناقصة (EtCO2/RR...) ويولد *_filled مثل MAP_filled.
    """
    df = df_in.copy()

    # ضروري لهذه الواجهة
    if "time" not in df.columns:
        df["time"] = np.arange(len(df), dtype=float)

    # حول الأعمدة الأساسية لأرقام
    base = ["time", "MAP", "HR", "SpO2", "RR", "EtCO2"]
    df = coerce_numeric(df, base)

    # جهّز filled للعلامات الأساسية
    # لو ناقص MAP أو HR أو SpO2 نخليها NaN
    if "MAP" not in df.columns:
        df["MAP"] = np.nan
    if "HR" not in df.columns:
        df["HR"] = np.nan
    if "SpO2" not in df.columns:
        df["SpO2"] = np.nan
    if "RR" not in df.columns:
        df["RR"] = np.nan
    if "EtCO2" not in df.columns:
        df["EtCO2"] = np.nan

    df["MAP_filled"] = df["MAP"].astype(float).ffill().bfill()
    df["HR_filled"] = df["HR"].astype(float).ffill().bfill()
    df["SpO2_filled"] = df["SpO2"].astype(float).ffill().bfill()
    df["RR_filled"] = df["RR"].astype(float).ffill().bfill()
    df["EtCO2_filled"] = df["EtCO2"].astype(float).ffill().bfill()

    # ابنِ X بنفس أعمدة التدريب
    X = pd.DataFrame(index=df.index)
    for col in expected_cols:
        if col in df.columns:
            X[col] = df[col]
        else:
            # أي عمود كان في التدريب لكنه غير موجود الآن -> NaN
            X[col] = np.nan

    # نفس ترتيب الأعمدة تماماً
    X = X[expected_cols]
    return X

st.title("🫀 Hypotension Early Warning Dashboard")
st.caption("Upload patient CSV → (Gate) → model → alarms")

# =========================
# Sidebar: Patient Info
# =========================
st.sidebar.header("🧾 Patient Summary")
patient_id = st.sidebar.text_input("🧑‍⚕️ Patient ID", value="P-001")
age = st.sidebar.number_input("🎂 Age", min_value=0, max_value=120, value=45, step=1)
sex = st.sidebar.selectbox("⚧ Sex", ["Male", "Female"])
unit = st.sidebar.selectbox("🏥 ICU / OR", ["ICU", "OR"])

drop_type = st.sidebar.selectbox(
    "اختيار نوع الهبوط",
    ["A: Rapid", "B: Gradual", "C: Intermittent"],
    index=0
)

st.sidebar.header("⚙️ إعدادات النموذج")
threshold = st.sidebar.slider("Threshold يدوي", 0.01, 0.99, 0.15, 0.01)
use_gate = st.sidebar.checkbox("Enable Gate", value=True)

st.sidebar.header("🧾 طريقة الإدخال")
input_mode = st.sidebar.radio("Input Mode", ["CSV Upload", "Manual Entry"], index=0)

df = None

if input_mode == "CSV Upload":
    uploaded_file = st.file_uploader("Upload patient CSV file", type=["csv"])
    st.info("CSV must contain at least: time, MAP, HR, SpO2 (RR optional, EtCO2 optional).")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

else:
    st.subheader("✍️ إدخال يدوي (بدون CSV)")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        t = st.number_input("time", value=0.0, step=1.0)
    with c2:
        MAP = st.number_input("MAP", value=80.0, step=1.0)
    with c3:
        HR = st.number_input("HR", value=78.0, step=1.0)
    with c4:
        SpO2 = st.number_input("SpO2", value=98.0, step=1.0)
    with c5:
        RR = st.number_input("RR", value=16.0, step=1.0)

    df = pd.DataFrame([{"time": t, "MAP": MAP, "HR": HR, "SpO2": SpO2, "RR": RR}])

if df is None:
    st.stop()

# =========================
# Inference
# =========================
def run_inference(df_in: pd.DataFrame, threshold: float, use_gate: bool):
    df_in = df_in.copy()

    # عرض القياسات الخام
    show_cols = [c for c in ["MAP", "HR", "SpO2", "RR", "EtCO2"] if c in df_in.columns]
    if show_cols:
        st.subheader("📈 Raw Vitals")
        st.line_chart(df_in[show_cols])

    expected_cols = get_expected_feature_names(model)
    if not expected_cols:
        raise ValueError("Model does not expose feature_names_in_. Re-upload a compatible model.joblib.")

    X = build_X_like_training(df_in, expected_cols)

    # Gate (يشتغل على df_in)
    if use_gate:
        gate_mask = apply_gate(df_in)
        gate_mask = gate_mask.reindex(df_in.index).fillna(False).astype(bool)
    else:
        gate_mask = pd.Series([True]*len(df_in), index=df_in.index)

    probs = model.predict_proba(X)[:, 1]
    probs = np.where(gate_mask.to_numpy(), probs, 0.0)

    df_out = df_in.copy()
    df_out["risk_score"] = probs
    df_out["alarm"] = df_out["risk_score"].apply(lambda r: generate_alarm(r, threshold))

    return df_out, gate_mask, expected_cols

try:
    df_out, gate_mask, expected_cols = run_inference(df, threshold=threshold, use_gate=use_gate)

    st.subheader("🚨 Alarm Timeline")
    st.line_chart(df_out[["risk_score"]])

    st.subheader("🩺 Current Status")
    latest = df_out.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAP", f"{float(latest.get('MAP', np.nan)):.1f}" if "MAP" in latest else "NA")
    c2.metric("Risk Score", f"{float(latest['risk_score']):.3f}")
    c3.metric("Alarm", "YES 🚨" if bool(latest["alarm"]) else "NO ✅")
    c4.metric("Drop Type", drop_type.split(":")[0].strip())

    st.subheader("🧾 Patient Info")
    st.write({
        "Patient ID": patient_id,
        "Age": int(age),
        "Sex": sex,
        "ICU/OR": unit,
        "Drop Type": drop_type
    })

    with st.expander("Show expected model columns"):
        st.write(expected_cols)

    with st.expander("Show output table"):
        st.dataframe(df_out)

except Exception as e:
    st.error("Error during inference:")
    st.exception(e)
