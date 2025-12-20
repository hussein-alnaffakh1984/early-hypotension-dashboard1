# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from features import build_feature_matrix, get_expected_feature_columns, compute_drop_scores
from gate import apply_gate
from alarm import generate_alarm

from explain import build_medical_explanation
from report_pdf import generate_pdf_report


st.set_page_config(page_title="Hypotension Early Warning Dashboard", layout="wide")
st.title("🫀 Hypotension Early Warning Dashboard")
st.caption("Upload patient CSV → features → (Gate) → model → alarms")


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


model = load_model()


def patch_simple_imputer(obj):
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


def t(lang_code: str, en: str, ar: str) -> str:
    return en if lang_code == "en" else ar


# ================= Sidebar =================
st.sidebar.header("🧾 Patient Summary")
patient_id = st.sidebar.text_input("🧑‍⚕️ Patient ID", value="P-001")
age = st.sidebar.number_input("🎂 Age", min_value=0, max_value=130, value=45, step=1)
sex = st.sidebar.selectbox("⚧ Sex", ["Male", "Female"])
location = st.sidebar.selectbox("🏥 ICU / OR", ["ICU", "OR"])
st.sidebar.divider()

st.sidebar.header("🌐 Language")
lang_ui = st.sidebar.radio("Explanation & Report", ["English", "العربية"], index=0)
lang_code = "en" if lang_ui == "English" else "ar"
st.sidebar.divider()

st.sidebar.header("⚙️ Model Settings")
threshold = st.sidebar.slider("Threshold (manual)", 0.01, 0.99, 0.11)
use_gate = st.sidebar.checkbox("Enable Gate", value=True)

# Drop selection includes AUTO
drop_type = st.sidebar.selectbox(
    t(lang_code, "Drop Mode", "وضع الهبوط"),
    ["AUTO (Research Goal)", "A: Rapid", "B: Gradual", "C: Intermittent"],
    index=0
)
st.sidebar.divider()

st.sidebar.header(t(lang_code, "Input Mode", "طريقة الإدخال"))
input_mode = st.sidebar.radio(t(lang_code, "Input Mode", "طريقة الإدخال"), ["CSV Upload", "Manual Entry"], index=0)


# ================= Helpers =================
def normalize_input_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    required = ["time", "MAP", "HR", "SpO2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    for col in ["RR", "EtCO2"]:
        if col not in df.columns:
            df[col] = np.nan

    for c in ["time", "MAP", "HR", "SpO2", "RR", "EtCO2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("time").reset_index(drop=True)
    return df


def align_features_to_expected(X: pd.DataFrame, expected_cols_list) -> pd.DataFrame:
    return X.reindex(columns=list(expected_cols_list), fill_value=np.nan)


def safe_apply_gate(X: pd.DataFrame, drop_key: str):
    out = apply_gate(X, drop_key=drop_key)
    if isinstance(out, tuple):
        if len(out) >= 2:
            return out[0], out[1]
        return out[0], None
    return out, None


def apply_drop_weighting(df_out: pd.DataFrame, scores_df: pd.DataFrame, mode: str):
    """
    Merge scores into df_out by time, and choose/weight based on mode.
    ✅ Fix: force BOTH time columns to float64 before merge_asof.
    """
    d = df_out.copy()
    s = scores_df.copy()

    # ✅ أهم سطرين لحل merge dtype
    d["time"] = pd.to_numeric(d["time"], errors="coerce").astype(np.float64)
    s["time"] = pd.to_numeric(s["time"], errors="coerce").astype(np.float64)

    d = d.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    s = s.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    d = pd.merge_asof(d, s, on="time", direction="nearest")

    # mode can be "AUTO" or "A"/"B"/"C"
    if mode == "AUTO":
        # pick best based on instantaneous highest score
        best = d[["score_A", "score_B", "score_C"]].values
        idx = np.argmax(best, axis=1)
        # map 0->A,1->B,2->C
        chosen = np.array(["A", "B", "C"])[idx]
        d["drop_auto"] = chosen
        # weight risk_score slightly by chosen score (optional)
        w = np.choose(idx, [d["score_A"], d["score_B"], d["score_C"]]).astype(np.float64)
        d["risk_score_weighted"] = (d["risk_score"] * (1.0 + 0.10 * w)).clip(0, 1)
    else:
        key = str(mode).strip().upper()
        if key not in ["A", "B", "C"]:
            key = "A"
        d["drop_auto"] = key
        w = d[f"score_{key}"].astype(np.float64)
        d["risk_score_weighted"] = (d["risk_score"] * (1.0 + 0.10 * w)).clip(0, 1)

    # decide alarm based on weighted score if exists
    if "risk_score_weighted" in d.columns:
        d["risk_score"] = d["risk_score_weighted"]

    return d



def run_inference(df_raw: pd.DataFrame, threshold: float, use_gate: bool, drop_mode: str):
    df = normalize_input_df(df_raw)

    # compute drop scores for AUTO and weighting
    scores_df = compute_drop_scores(df)

    # choose drop_key for gate:
    if drop_mode == "AUTO":
        # gate by the dominant type overall (case-level)
        counts = scores_df["drop_auto"].value_counts()
        drop_key = counts.index[0] if len(counts) else "A"
    else:
        drop_key = drop_mode

    # features
    X = build_feature_matrix(df, expected_cols=expected_cols)
    if use_gate:
        X, gate_mask = safe_apply_gate(X, drop_key=drop_key)
    else:
        gate_mask = None

    X = align_features_to_expected(X, expected_cols)
    probs = model.predict_proba(X)[:, 1]

    df_out = df.copy()
    df_out["risk_score"] = probs
    df_out["alarm"] = df_out["risk_score"].apply(lambda s: generate_alarm(s, threshold))

    # apply weighting so A/B/C differ
    df_out = apply_drop_weighting(df_out, scores_df, mode=("AUTO" if drop_mode == "AUTO" else drop_key))
    df_out["alarm"] = df_out["risk_score"].apply(lambda s: generate_alarm(s, threshold))

    return df_out, gate_mask, scores_df, drop_key


def compare_drop_types(df_raw: pd.DataFrame, threshold: float, use_gate: bool):
    rows = []
    for key, label in [("A", "A: Rapid"), ("B", "B: Gradual"), ("C", "C: Intermittent"), ("AUTO", "AUTO")]:
        try:
            df_out, _, scores_df, chosen = run_inference(df_raw, threshold=threshold, use_gate=use_gate, drop_mode=key)
            last = df_out.iloc[-1]
            rows.append({
                "Mode": label,
                "GateKey": chosen,
                "Last MAP": float(last["MAP"]),
                "Last Risk": float(last["risk_score"]),
                "Alarm": "YES 🚨" if bool(last["alarm"]) else "NO ✅",
            })
        except Exception as e:
            rows.append({"Mode": label, "GateKey": "-", "Last MAP": np.nan, "Last Risk": np.nan, "Alarm": f"ERROR: {e}"})
    return pd.DataFrame(rows)


# ================= Main UI =================
df_input = None

if input_mode == "CSV Upload":
    uploaded_file = st.file_uploader(t(lang_code, "Upload patient CSV file", "رفع ملف CSV للمريض"), type=["csv"])
    st.info(t(lang_code,
              "CSV must contain at least: time, MAP, HR, SpO2 (RR optional, EtCO2 optional).",
              "يجب أن يحتوي CSV على الأقل: time, MAP, HR, SpO2 (RR اختياري و EtCO2 اختياري)."))
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
else:
    st.subheader(t(lang_code, "🧾 Manual Entry", "🧾 إدخال يدوي"))
    n_points = st.number_input(t(lang_code, "Number of points", "عدد النقاط"), 1, 600, 60, 1)
    start_time = st.number_input(t(lang_code, "Start time", "زمن البداية"), value=0.0)
    step_time = st.number_input(t(lang_code, "Time step (sec)", "فاصل الزمن (ثانية)"), value=1.0)

    map_start = st.number_input("MAP start", value=85.0)
    map_end = st.number_input("MAP end", value=55.0)
    hr_start = st.number_input("HR start", value=75.0)
    hr_end = st.number_input("HR end", value=105.0)
    spo2_start = st.number_input("SpO2 start", value=98.0)
    spo2_end = st.number_input("SpO2 end", value=93.0)
    rr_start = st.number_input("RR start (optional)", value=16.0)
    rr_end = st.number_input("RR end (optional)", value=24.0)

    if st.button(t(lang_code, "Generate", "توليد")):
        t_arr = np.arange(n_points, dtype=float) * float(step_time) + float(start_time)
        df_input = pd.DataFrame({
            "time": t_arr,
            "MAP": np.linspace(map_start, map_end, n_points),
            "HR": np.linspace(hr_start, hr_end, n_points),
            "SpO2": np.linspace(spo2_start, spo2_end, n_points),
            "RR": np.linspace(rr_start, rr_end, n_points),
        })

if df_input is None:
    st.info(t(lang_code, "⬅️ Choose input method and provide data.", "⬅️ اختر طريقة الإدخال ثم وفّر بيانات."))
    st.stop()

try:
    df_norm = normalize_input_df(df_input)

    patient_info = {
        "Patient ID": patient_id,
        "Age": age,
        "Sex": sex,
        "ICU/OR": location,
        "Drop Type": drop_type
    }

    st.subheader(t(lang_code, "📈 Raw Vitals", "📈 الحيويات الخام"))
    st.line_chart(df_norm[["MAP", "HR", "SpO2"] + (["RR"] if "RR" in df_norm.columns else [])])

    # mode mapping
    if drop_type.startswith("AUTO"):
        mode = "AUTO"
        drop_text = "AUTO"
    else:
        mode = drop_type.split(":")[0].strip()
        drop_text = drop_type

    df_out, gate_mask, scores_df, gate_key_used = run_inference(df_norm, threshold=threshold, use_gate=use_gate, drop_mode=mode)

    st.subheader(t(lang_code, "🚨 Alarm Timeline", "🚨 خط الإنذار الزمني"))
    st.line_chart(df_out[["risk_score"]])

    latest = df_out.iloc[-1]
    st.subheader(t(lang_code, "🩺 Current Status", "🩺 الحالة الحالية"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAP", f"{latest['MAP']:.1f}")
    c2.metric(t(lang_code, "Risk Score", "درجة الخطر"), f"{latest['risk_score']:.3f}")
    c3.metric(t(lang_code, "Alarm", "إنذار"), "YES 🚨" if latest["alarm"] else "NO ✅")
    c4.metric(t(lang_code, "Gate Key Used", "نوع Gate المستخدم"), gate_key_used)

    # show AUTO drop over time
    st.subheader(t(lang_code, "🧭 Auto Drop-Type (Research)", "🧭 نوع الهبوط تلقائيًا (بحثيًا)"))
    st.caption(t(lang_code, "AUTO is derived from signal morphology (rapid/gradual/intermittent).",
                 "AUTO يُستنتج من شكل الإشارة (سريع/تدريجي/متقطع)."))
    st.dataframe(scores_df.tail(15), use_container_width=True)

    # explanation
    st.subheader(t(lang_code, "🧠 Medical Explanation (auto)", "🧠 تفسير طبي (آلي)"))
    exp = build_medical_explanation(df_out, threshold=threshold, drop_key=gate_key_used, use_gate=use_gate, lang=lang_code)
    st.success(exp["headline"]) if not latest["alarm"] else st.error(exp["headline"])
    st.markdown(f"**{exp['reasons_title']}**")
    for r in exp["reasons"]:
        st.write("•", r)
    st.markdown(f"**{exp['rec_title']}**")
    for r in exp["recommendation"]:
        st.write("•", r)
    st.caption(exp["disclaimer"])

    # PDF
    st.subheader(t(lang_code, "📄 PDF Report", "📄 تقرير PDF"))
    pdf_bytes = generate_pdf_report(df_out, patient_info, exp, threshold, drop_text, lang=lang_code)
    st.download_button(
        t(lang_code, "⬇️ Download PDF Report", "⬇️ تحميل تقرير PDF"),
        data=pdf_bytes,
        file_name=f"{patient_id}_report.pdf",
        mime="application/pdf"
    )

    # Compare modes
    st.subheader(t(lang_code, "🔁 Compare A / B / C / AUTO", "🔁 مقارنة A / B / C / AUTO"))
    comp = compare_drop_types(df_norm, threshold=threshold, use_gate=use_gate)
    st.dataframe(comp, use_container_width=True)

    # Download output
    st.download_button(
        t(lang_code, "⬇️ Download output CSV", "⬇️ تحميل نتائج CSV"),
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name=f"{patient_id}_output.csv",
        mime="text/csv"
    )

    with st.expander(t(lang_code, "Show expected model columns", "إظهار أعمدة النموذج المتوقعة")):
        st.write(list(expected_cols))

except Exception as e:
    st.error(t(lang_code, "Error during inference:", "خطأ أثناء الاستدلال:"))
    st.exception(e)
