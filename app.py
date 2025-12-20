# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from features import build_feature_matrix, get_expected_feature_columns
from gate import apply_gate
from alarm import generate_alarm

from explain import build_medical_explanation
from report_pdf import generate_pdf_report


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
    return joblib.load("model.joblib")

model = load_model()


def patch_simple_imputer(obj):
    """
    Fix for: AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'
    Happens due to sklearn version mismatch between training vs runtime.
    """
    if isinstance(obj, SimpleImputer):
        if not hasattr(obj, "_fill_dtype"):
            obj._fill_dtype = np.float64
        return

    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            patch_simple_imputer(step)
        return

    if isinstance(obj, ColumnTransformer):
        for _, trans, _ in obj.transformers:
            if trans in ("drop", "passthrough"):
                continue
            patch_simple_imputer(trans)

        rem = getattr(obj, "remainder", None)
        if rem not in (None, "drop", "passthrough"):
            patch_simple_imputer(rem)
        return

    if hasattr(obj, "get_params"):
        for v in obj.get_params(deep=False).values():
            if hasattr(v, "__class__"):
                patch_simple_imputer(v)


patch_simple_imputer(model)
expected_cols = get_expected_feature_columns(model)


# ===============================
# Language helper
# ===============================
def t(lang_code: str, en: str, ar: str) -> str:
    return en if lang_code == "en" else ar


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

# Drop type: Auto + manual A/B/C
drop_type = st.sidebar.selectbox(
    "Drop type",
    ["Auto", "A: Rapid", "B: Gradual", "C: Intermittent"],
    index=0
)
drop_key = None
drop_text = drop_type
if drop_type != "Auto":
    drop_key = drop_type.split(":")[0].strip().upper()
    drop_text = {"A": "A: Rapid", "B": "B: Gradual", "C": "C: Intermittent"}.get(drop_key, drop_type)

st.sidebar.divider()

# ===============================
# Sidebar: Language
# ===============================
st.sidebar.header("🌐 Language")
lang_ui = st.sidebar.radio("Explanation & Report", ["English", "العربية"], index=0)
lang_code = "en" if lang_ui == "English" else "ar"
st.sidebar.divider()

# ===============================
# Sidebar: Input Mode
# ===============================
st.sidebar.header(t(lang_code, "Input Mode", "طريقة الإدخال"))
input_mode = st.sidebar.radio(t(lang_code, "Input Mode", "طريقة الإدخال"), ["CSV Upload", "Manual Entry"], index=0)


# ===============================
# Helpers
# ===============================
def normalize_input_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    required = ["time", "MAP", "HR", "SpO2"]
    missing_req = [c for c in required if c not in df.columns]
    if missing_req:
        raise ValueError(f"CSV is missing required columns: {missing_req}")

    if "RR" not in df.columns:
        df["RR"] = np.nan
    if "EtCO2" not in df.columns:
        df["EtCO2"] = np.nan

    for c in ["time", "MAP", "HR", "SpO2", "RR", "EtCO2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("time").reset_index(drop=True)
    return df


def align_features_to_expected(X: pd.DataFrame, expected_cols_list) -> pd.DataFrame:
    X = X.copy()
    X = X.reindex(columns=list(expected_cols_list), fill_value=np.nan)
    return X


def safe_apply_gate(X: pd.DataFrame, drop_key_used: str):
    out = apply_gate(X, drop_key=drop_key_used)

    # expected: (X, mask)
    if isinstance(out, tuple):
        if len(out) >= 2:
            return out[0], out[1]
        if len(out) == 1:
            return out[0], None
        return X, None

    # if mask only
    if isinstance(out, (pd.Series, np.ndarray, list)) and not isinstance(out, pd.DataFrame):
        return X, np.asarray(out, dtype=bool)

    # X only
    return out, None


def auto_drop_type_from_map(df: pd.DataFrame) -> str:
    """
    Auto select A/B/C from MAP pattern (simple heuristic).
    """
    d = df[["time", "MAP"]].dropna().copy()
    if len(d) < 6:
        return "B"

    dt = np.median(np.diff(d["time"].values)) if len(d) > 2 else 1.0
    dt = 1.0 if not np.isfinite(dt) or dt <= 0 else float(dt)

    w2 = max(2, int(round(2.0 / dt)))
    w8 = max(4, int(round(8.0 / dt)))

    mapv = d["MAP"].values.astype(float)

    if len(mapv) > w2:
        delta2 = mapv[w2:] - mapv[:-w2]
        worst_drop_2m = np.min(delta2)
    else:
        worst_drop_2m = 0.0

    # overall slope
    denom = max(1e-9, (d["time"].values[-1] - d["time"].values[0]))
    overall_slope = (mapv[-1] - mapv[0]) / denom

    # sign changes in first differences (proxy for intermittent)
    diff1 = np.diff(mapv)
    sign = np.sign(diff1)
    sign_changes = np.sum(sign[1:] * sign[:-1] < 0)

    std = np.nanstd(mapv)

    if worst_drop_2m <= -15:
        return "A"
    if sign_changes >= 4 and std >= 5:
        return "C"
    if overall_slope < -0.5:
        return "B"
    return "B"


def run_inference(df_raw: pd.DataFrame, threshold: float, use_gate: bool, drop_key_in: str | None):
    df = normalize_input_df(df_raw)

    # Auto drop type if needed
    auto_used = False
    if drop_key_in is None:
        drop_key_used = auto_drop_type_from_map(df)
        auto_used = True
    else:
        drop_key_used = drop_key_in.strip().upper()

    # 1) Extract features
    X = build_feature_matrix(df, expected_cols=expected_cols)

    # 2) Gate
    gate_mask = None
    if use_gate:
        X, gate_mask = safe_apply_gate(X, drop_key_used=drop_key_used)

    # 3) Align columns
    X = align_features_to_expected(X, expected_cols)

    # 4) Predict
    probs = model.predict_proba(X)[:, 1]

    # 5) APPLY GATE TO PROBABILITIES (CRITICAL)
    if gate_mask is not None:
        gate_mask = np.asarray(gate_mask, dtype=bool)
        if len(gate_mask) == len(probs):
            probs = probs * gate_mask.astype(float)

    # 6) Output
    df_out = df.copy()
    df_out["risk_score"] = probs
    df_out["alarm"] = df_out["risk_score"].apply(lambda s: generate_alarm(s, threshold))

    return df_out, gate_mask, X, drop_key_used, auto_used


def compare_drop_types(df_raw: pd.DataFrame, threshold: float, use_gate: bool):
    rows = []
    for key, label in [("A", "A: Rapid"), ("B", "B: Gradual"), ("C", "C: Intermittent")]:
        try:
            df_out, gate_mask, _, used_key, _ = run_inference(df_raw, threshold, use_gate, drop_key_in=key)
            last = df_out.iloc[-1]
            gate_last = None
            if gate_mask is not None and len(gate_mask) == len(df_out):
                gate_last = bool(gate_mask[-1])

            rows.append({
                "Drop Type": label,
                "Last MAP": float(last["MAP"]),
                "Last Risk": float(last["risk_score"]),
                "Alarm": "YES 🚨" if bool(last["alarm"]) else "NO ✅",
                "Gate(last)": gate_last
            })
        except Exception as e:
            rows.append({
                "Drop Type": label,
                "Last MAP": np.nan,
                "Last Risk": np.nan,
                "Alarm": f"ERROR: {e}",
                "Gate(last)": None
            })
    return pd.DataFrame(rows)


# ===============================
# Main UI
# ===============================
df_input = None

if input_mode == "CSV Upload":
    uploaded_file = st.file_uploader(t(lang_code, "Upload patient CSV file", "رفع ملف CSV للمريض"), type=["csv"])
    st.info(t(lang_code,
              "CSV must contain at least: time, MAP, HR, SpO2 (RR & EtCO2 optional).",
              "يجب أن يحتوي CSV على الأقل: time, MAP, HR, SpO2 (و RR و EtCO2 اختياري)."))
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)

else:
    st.subheader(t(lang_code, "🧾 Manual Entry", "🧾 إدخال يدوي"))
    st.caption(t(lang_code,
                 "Generate a synthetic time series for quick testing.",
                 "توليد سلسلة زمنية تجريبية للاختبار السريع."))

    n_points = st.number_input(t(lang_code, "Number of time points", "عدد النقاط الزمنية"),
                               min_value=6, max_value=300, value=16, step=1)

    colA, colB = st.columns(2)
    with colA:
        start_time = st.number_input(t(lang_code, "Start time", "زمن البداية"), value=0.0)
        step_time = st.number_input(t(lang_code, "Time step", "فاصل الزمن"), value=1.0)
    with colB:
        map_start = st.number_input("MAP start", value=82.0)
        map_end = st.number_input("MAP end", value=56.0)

    hr_start = st.number_input("HR start", value=78.0)
    hr_end = st.number_input("HR end", value=110.0)
    spo2_start = st.number_input("SpO2 start", value=98.0)
    spo2_end = st.number_input("SpO2 end", value=91.0)
    rr_start = st.number_input("RR start (optional)", value=16.0)
    rr_end = st.number_input("RR end (optional)", value=28.0)
    et_start = st.number_input("EtCO2 start (optional)", value=36.0)
    et_end = st.number_input("EtCO2 end (optional)", value=30.0)

    if st.button(t(lang_code, "Generate Manual Timeseries", "توليد سلسلة زمنية يدويًا")):
        t_arr = np.arange(n_points, dtype=float) * float(step_time) + float(start_time)
        df_input = pd.DataFrame({
            "time": t_arr,
            "MAP": np.linspace(map_start, map_end, n_points),
            "HR": np.linspace(hr_start, hr_end, n_points),
            "SpO2": np.linspace(spo2_start, spo2_end, n_points),
            "RR": np.linspace(rr_start, rr_end, n_points),
            "EtCO2": np.linspace(et_start, et_end, n_points),
        })


if df_input is None:
    st.info(t(lang_code, "⬅️ Choose an input method and provide data.", "⬅️ اختر طريقة إدخال ثم وفّر بيانات."))
    st.stop()


try:
    df_norm = normalize_input_df(df_input)

    patient_info = {
        "Patient ID": patient_id,
        "Age": age,
        "Sex": sex,
        "ICU/OR": location,
        "Drop Type": drop_text
    }

    st.subheader(t(lang_code, "📈 Raw Vitals", "📈 الحيويات الخام"))
    chart_cols = ["HR", "MAP", "SpO2"]
    if "RR" in df_norm.columns:
        chart_cols.append("RR")
    if "EtCO2" in df_norm.columns:
        chart_cols.append("EtCO2")
    st.line_chart(df_norm[chart_cols])

    # Inference
    df_out, gate_mask, X, used_drop_key, auto_used = run_inference(
        df_norm, threshold=threshold, use_gate=use_gate, drop_key_in=drop_key
    )

    if auto_used:
        st.info(t(lang_code,
                  f"Auto detected drop type: {used_drop_key}",
                  f"تم تحديد نوع الهبوط تلقائياً: {used_drop_key}"))

    st.subheader(t(lang_code, "🚨 Alarm Timeline", "🚨 خط الإنذار الزمني"))
    st.line_chart(df_out[["risk_score"]])

    latest = df_out.iloc[-1]
    st.subheader(t(lang_code, "🩺 Current Status", "🩺 الحالة الحالية"))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAP", f"{latest['MAP']:.1f}")
    c2.metric(t(lang_code, "Risk Score", "درجة الخطر"), f"{latest['risk_score']:.3f}")
    c3.metric(t(lang_code, "Alarm", "إنذار"), "YES 🚨" if latest["alarm"] else "NO ✅")
    c4.metric(t(lang_code, "Drop Type", "نوع الهبوط"), used_drop_key)

    gate_last = None
    if gate_mask is not None and len(gate_mask) == len(df_out):
        gate_last = bool(gate_mask[-1])
    c5.metric("Gate(last)", str(gate_last))

    # Explanation
    st.subheader(t(lang_code, "🧠 Medical Explanation (auto)", "🧠 تفسير طبي (آلي)"))
    exp = build_medical_explanation(
        df_out,
        threshold=threshold,
        drop_key=used_drop_key,
        use_gate=use_gate,
        lang=lang_code
    )

    if latest["alarm"]:
        st.error(exp["headline"])
    else:
        st.success(exp["headline"])

    st.markdown(f"**{exp.get('reasons_title', t(lang_code,'Why?','لماذا؟'))}**")
    for r in exp["reasons"]:
        st.write("•", r)

    st.markdown(f"**{exp.get('rec_title', t(lang_code,'Recommendation','التوصيات'))}**")
    for r in exp["recommendation"]:
        st.write("•", r)

    st.caption(exp["disclaimer"])

    # PDF report
    st.subheader(t(lang_code, "📄 PDF Report", "📄 تقرير PDF"))
    pdf_bytes = generate_pdf_report(
        df_out=df_out,
        patient_info=patient_info,
        explanation=exp,
        threshold=threshold,
        drop_text=(drop_text if drop_type != "Auto" else f"Auto → {used_drop_key}"),
        lang=lang_code
    )
    st.download_button(
        t(lang_code, "⬇️ Download PDF Report", "⬇️ تحميل تقرير PDF"),
        data=pdf_bytes,
        file_name=f"{patient_id}_report.pdf",
        mime="application/pdf"
    )

    # Debug + Transparency
    with st.expander(t(lang_code, "Show expected model columns", "إظهار أعمدة النموذج المتوقعة")):
        st.write(list(expected_cols))

    with st.expander(t(lang_code, "Show extracted feature matrix (head)", "إظهار أول صفوف مصفوفة الخصائص")):
        st.dataframe(X.head(10), use_container_width=True)

    # Compare A/B/C (FOR RESEARCH)
    st.subheader(t(lang_code, "🔁 Compare A / B / C (same data)", "🔁 مقارنة A / B / C (نفس البيانات)"))
    comp_df = compare_drop_types(df_norm, threshold=threshold, use_gate=use_gate)
    st.dataframe(comp_df, use_container_width=True)

    # Download CSV
    st.download_button(
        t(lang_code, "⬇️ Download output CSV (with risk/alarm)", "⬇️ تحميل نتائج CSV (الخطر/الإنذار)"),
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name=f"{patient_id}_output.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error(t(lang_code, "Error during inference:", "خطأ أثناء الاستدلال:"))
    st.exception(e)
