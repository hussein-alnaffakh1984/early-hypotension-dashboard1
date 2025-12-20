import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from features import (
    build_feature_matrix,
    get_expected_feature_columns,
)

# optional: if your features.py contains these, we'll use them
try:
    from features import compute_drop_scores
except Exception:
    compute_drop_scores = None

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
# Load model
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


model = load_model()


# ===============================
# Fix sklearn SimpleImputer mismatch
# ===============================
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
# Sidebar: Language
# ===============================
st.sidebar.header("🌐 Language")
lang_ui = st.sidebar.radio("Explanation & Report", ["English", "العربية"], index=0)
lang_code = "en" if lang_ui == "English" else "ar"


# ===============================
# Sidebar: Patient Info
# ===============================
st.sidebar.header(t(lang_code, "🧾 Patient Summary", "🧾 معلومات المريض"))
patient_id = st.sidebar.text_input(t(lang_code, "Patient ID", "رقم المريض"), value="P-001")
age = st.sidebar.number_input(t(lang_code, "Age", "العمر"), min_value=0, max_value=130, value=45, step=1)
sex = st.sidebar.selectbox(t(lang_code, "Sex", "الجنس"), ["Male", "Female"])
location = st.sidebar.selectbox(t(lang_code, "ICU / OR", "ICU / OR"), ["ICU", "OR"])
st.sidebar.divider()


# ===============================
# Sidebar: Model Settings
# ===============================
st.sidebar.header(t(lang_code, "⚙️ Model Settings", "⚙️ إعدادات النموذج"))
threshold = st.sidebar.slider(t(lang_code, "Threshold (alarm decision)", "العتبة (قرار الإنذار)"), 0.01, 0.99, 0.11)
use_gate = st.sidebar.checkbox(t(lang_code, "Enable Gate", "تفعيل Gate"), value=True)

mode = st.sidebar.selectbox(
    t(lang_code, "Drop Mode", "نمط الهبوط"),
    ["AUTO", "A", "B", "C"],
    index=0
)
st.sidebar.caption(t(
    lang_code,
    "AUTO selects A/B/C automatically from signal behavior.",
    "AUTO يختار A/B/C تلقائيًا حسب سلوك الإشارة."
))
st.sidebar.divider()


# ===============================
# Sidebar: Input Mode
# ===============================
st.sidebar.header(t(lang_code, "📥 Input Mode", "📥 طريقة الإدخال"))
input_mode = st.sidebar.radio(t(lang_code, "Input", "الإدخال"), ["CSV Upload", "Manual Entry"], index=0)


# ===============================
# Helpers: normalize input
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

    # numeric
    for c in ["time", "MAP", "HR", "SpO2", "RR", "EtCO2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # IMPORTANT: time dtype unified as float for merge_asof
    df["time"] = df["time"].astype(float)

    df = df.sort_values("time").reset_index(drop=True)
    return df


def align_features_to_expected(X: pd.DataFrame, expected_cols_list) -> pd.DataFrame:
    X = X.copy()
    return X.reindex(columns=list(expected_cols_list), fill_value=np.nan)


def safe_apply_gate(X: pd.DataFrame, drop_key: str):
    """
    apply_gate may return:
      - X
      - (X, mask)
      - (X, mask, extra)
    """
    out = apply_gate(X, drop_key=drop_key)
    if isinstance(out, tuple):
        if len(out) == 0:
            return X, None
        if len(out) == 1:
            return out[0], None
        return out[0], out[1]
    return out, None


def decide_alarm(score: float, thr: float) -> bool:
    """
    Make threshold actually affect alarm decision.
    If generate_alarm exists, we keep it; otherwise fallback to score>=thr.
    """
    try:
        return bool(generate_alarm(float(score), float(thr)))
    except Exception:
        return float(score) >= float(thr)


# ===============================
# AUTO drop selection
# ===============================
def compute_auto_drop(df: pd.DataFrame):
    """
    Returns:
      scores_df (may be None)
      auto_key ("A"/"B"/"C")
    Strategy:
      - If features.compute_drop_scores exists and returns drop_auto -> use it
      - Else fallback to heuristic based on MAP slope/variability
    """
    scores_df = None

    # 1) Try compute_drop_scores if available
    if compute_drop_scores is not None:
        try:
            scores_df = compute_drop_scores(df)
            if scores_df is not None and len(scores_df) > 0:
                if "time" in scores_df.columns:
                    scores_df = scores_df.copy()
                    scores_df["time"] = pd.to_numeric(scores_df["time"], errors="coerce").astype(float)
                    scores_df = scores_df.sort_values("time").reset_index(drop=True)

                # If drop_auto exists
                if "drop_auto" in scores_df.columns:
                    auto_key = str(scores_df["drop_auto"].iloc[-1]).strip().upper()
                    if auto_key in ["A", "B", "C"]:
                        return scores_df, auto_key

                # else if have drop_A/drop_B/drop_C
                cand = [c for c in ["drop_A", "drop_B", "drop_C"] if c in scores_df.columns]
                if len(cand) == 3:
                    winners = scores_df[cand].idxmax(axis=1).str.replace("drop_", "", regex=False)
                    auto_key = winners.value_counts().idxmax()
                    if auto_key in ["A", "B", "C"]:
                        return scores_df, auto_key
        except Exception:
            scores_df = None

    # 2) Fallback heuristic (works always)
    # Use last 10% segment behavior
    d = df.copy()
    d = d.dropna(subset=["MAP"]).reset_index(drop=True)
    if len(d) < 4:
        return scores_df, "C"

    n = len(d)
    k = max(4, int(0.25 * n))
    seg = d.tail(k)

    # slope of MAP
    x = seg["time"].values
    y = seg["MAP"].values
    if np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and len(np.unique(x)) > 1:
        slope = np.polyfit(x, y, 1)[0]
    else:
        slope = 0.0

    # variability
    std_map = float(np.nanstd(seg["MAP"].values))

    # interpret:
    # A (rapid): strong negative slope
    # B (gradual): mild negative slope
    # C (intermittent): high variability / oscillations
    if std_map >= 6.0:
        return scores_df, "C"
    if slope <= -2.0:
        return scores_df, "A"
    if slope <= -0.5:
        return scores_df, "B"
    return scores_df, "C"


# ===============================
# Inference (robust to resampling lengths)
# ===============================
def run_inference(df_raw: pd.DataFrame, threshold: float, use_gate: bool, drop_mode: str):
    """
    Returns:
      df_out       : SAME length as original df (user input)
      gate_mask    : optional
      scores_df    : optional (may be resampled)
      gate_key_used: "A"/"B"/"C"
    """
    df = normalize_input_df(df_raw)

    # AUTO drop detection
    scores_df, auto_key = compute_auto_drop(df)
    gate_key_used = auto_key if drop_mode == "AUTO" else drop_mode

    # Feature extraction
    X = build_feature_matrix(df, expected_cols=expected_cols)

    # Attach time column if missing
    if "time" not in X.columns:
        X = X.copy()
        if len(X) == len(df):
            X["time"] = df["time"].values
        else:
            # synthetic time axis
            t0 = float(df["time"].iloc[0]) if len(df) else 0.0
            X["time"] = np.arange(len(X), dtype=float) + t0

    X["time"] = pd.to_numeric(X["time"], errors="coerce").astype(float)
    X = X.sort_values("time").reset_index(drop=True)

    # Gate (optional) on features only
    gate_mask = None
    if use_gate:
        X_feat = X.drop(columns=["time"], errors="ignore")
        X_feat, gate_mask = safe_apply_gate(X_feat, drop_key=gate_key_used)
        X = pd.concat([X[["time"]], X_feat], axis=1)

    # Align model features
    X_model = X.drop(columns=["time"], errors="ignore")
    X_model = align_features_to_expected(X_model, expected_cols)

    # Predict (can be longer due to internal resampling)
    probs = model.predict_proba(X_model)[:, 1]

    probs_df = pd.DataFrame({"time": X["time"].values, "risk_score": probs}).sort_values("time").reset_index(drop=True)

    # Map risk_score back to original df rows using merge_asof (fix length mismatch)
    df_out = df.copy().sort_values("time").reset_index(drop=True)
    df_out = pd.merge_asof(df_out, probs_df, on="time", direction="nearest")

    # Alarm decision depends on threshold
    df_out["alarm"] = df_out["risk_score"].apply(lambda s: decide_alarm(s, threshold))

    return df_out, gate_mask, scores_df, gate_key_used


def compare_abc(df_raw: pd.DataFrame, threshold: float, use_gate: bool):
    rows = []
    for key, label in [("A", "A: Rapid"), ("B", "B: Gradual"), ("C", "C: Intermittent")]:
        try:
            df_out, _, _, used = run_inference(df_raw, threshold, use_gate, drop_mode=key)
            last = df_out.iloc[-1]
            rows.append({
                "Drop Type": label,
                "Used": used,
                "Last MAP": float(last["MAP"]),
                "Last Risk": float(last["risk_score"]),
                "Alarm (thr)": "YES 🚨" if bool(last["alarm"]) else "NO ✅"
            })
        except Exception as e:
            rows.append({
                "Drop Type": label,
                "Used": key,
                "Last MAP": np.nan,
                "Last Risk": np.nan,
                "Alarm (thr)": f"ERROR: {e}"
            })
    return pd.DataFrame(rows)


# ===============================
# Main UI input
# ===============================
df_input = None

if input_mode == "CSV Upload":
    uploaded_file = st.file_uploader(t(lang_code, "Upload patient CSV file", "رفع ملف CSV"), type=["csv"])
    st.info(t(
        lang_code,
        "CSV must contain at least: time, MAP, HR, SpO2 (RR/EtCO2 optional).",
        "يجب أن يحتوي CSV على الأقل: time, MAP, HR, SpO2 (و RR/EtCO2 اختياري)."
    ))
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
else:
    st.subheader(t(lang_code, "🧾 Manual Entry", "🧾 إدخال يدوي"))
    st.caption(t(lang_code, "Generate a synthetic time series for testing.", "توليد سلسلة زمنية تجريبية للاختبار."))

    n_points = st.number_input(t(lang_code, "Number of points", "عدد النقاط"), min_value=4, max_value=1200, value=16, step=1)

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

    if st.button(t(lang_code, "Generate", "توليد")):
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
    st.info(t(lang_code, "⬅️ Provide input data to start.", "⬅️ وفّر بيانات للبدء."))
    st.stop()


# ===============================
# Run
# ===============================
try:
    df_norm = normalize_input_df(df_input)

    patient_info = {
        "Patient ID": patient_id,
        "Age": age,
        "Sex": sex,
        "ICU/OR": location,
        "Drop Mode": mode,
    }

    st.subheader(t(lang_code, "📈 Raw Vitals", "📈 الحيويات الخام"))
    chart_cols = ["MAP", "HR", "SpO2", "RR", "EtCO2"]
    show_cols = [c for c in chart_cols if c in df_norm.columns]
    st.line_chart(df_norm[show_cols])

    df_out, gate_mask, scores_df, gate_used = run_inference(
        df_norm, threshold=threshold, use_gate=use_gate, drop_mode=mode
    )

    st.subheader(t(lang_code, "🚨 Risk Timeline", "🚨 خطورة (زمنيًا)"))
    st.line_chart(df_out[["risk_score"]])

    latest = df_out.iloc[-1]
    st.subheader(t(lang_code, "🩺 Current Status", "🩺 الحالة الحالية"))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAP", f"{latest['MAP']:.1f}")
    c2.metric(t(lang_code, "Risk Score", "درجة الخطر"), f"{latest['risk_score']:.3f}")
    c3.metric(t(lang_code, "Alarm (by threshold)", "الإنذار (حسب العتبة)"), "YES 🚨" if latest["alarm"] else "NO ✅")
    c4.metric(t(lang_code, "Drop Mode", "نمط الهبوط"), mode)
    c5.metric(t(lang_code, "Auto/Used", "المختار تلقائياً/المستخدم"), gate_used)

    # Explanation
    st.subheader(t(lang_code, "🧠 Medical Explanation (auto)", "🧠 تفسير طبي (آلي)"))
    exp = build_medical_explanation(
        df_out=df_out,
        threshold=threshold,
        drop_key=gate_used,
        use_gate=use_gate,
        lang=lang_code
    )

    if latest["alarm"]:
        st.error(exp.get("headline", "Alarm triggered"))
    else:
        st.success(exp.get("headline", "No alarm"))

    st.markdown(f"**{exp.get('reasons_title', t(lang_code,'Why?','لماذا؟'))}**")
    for r in exp.get("reasons", []):
        st.write("•", r)

    st.markdown(f"**{exp.get('rec_title', t(lang_code,'Recommendation','التوصيات'))}**")
    for r in exp.get("recommendation", []):
        st.write("•", r)

    st.caption(exp.get("disclaimer", ""))

    # PDF report
    st.subheader(t(lang_code, "📄 PDF Report", "📄 تقرير PDF"))
    pdf_bytes = generate_pdf_report(
        df_out=df_out,
        patient_info=patient_info,
        explanation=exp,
        threshold=threshold,
        drop_text=f"{gate_used}",
        lang=lang_code
    )
    st.download_button(
        t(lang_code, "⬇️ Download PDF Report", "⬇️ تحميل تقرير PDF"),
        data=pdf_bytes,
        file_name=f"{patient_id}_report.pdf",
        mime="application/pdf"
    )

    # Compare A/B/C
    st.subheader(t(lang_code, "🔁 Compare A / B / C (same data)", "🔁 مقارنة A / B / C (نفس البيانات)"))
    comp_df = compare_abc(df_norm, threshold=threshold, use_gate=use_gate)
    st.dataframe(comp_df, use_container_width=True)

    # Debug panels
    with st.expander(t(lang_code, "Show expected model columns", "إظهار أعمدة النموذج المتوقعة")):
        st.write(list(expected_cols))

    if scores_df is not None:
        with st.expander(t(lang_code, "Show AUTO drop scores (head)", "عرض درجات AUTO (أول صفوف)")):
            st.dataframe(scores_df.head(20), use_container_width=True)

    # Download CSV output
    st.download_button(
        t(lang_code, "⬇️ Download output CSV", "⬇️ تحميل CSV النتائج"),
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name=f"{patient_id}_output.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error(t(lang_code, "Error during inference:", "خطأ أثناء الاستدلال:"))
    st.exception(e)
