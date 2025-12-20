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
    compute_drop_scores,          # لازم تكون موجودة في features.py
)
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

# expected feature columns exactly as trained (from model)
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
st.sidebar.divider()


# ===============================
# Sidebar: Patient Info
# ===============================
st.sidebar.header(t(lang_code, "🧾 Patient Summary", "🧾 معلومات المريض"))
patient_id = st.sidebar.text_input(t(lang_code, "🧑‍⚕️ Patient ID", "🧑‍⚕️ رقم المريض"), value="P-001")
age = st.sidebar.number_input(t(lang_code, "🎂 Age", "🎂 العمر"), min_value=0, max_value=130, value=45, step=1)
sex = st.sidebar.selectbox(t(lang_code, "⚧ Sex", "⚧ الجنس"), ["Male", "Female"])
location = st.sidebar.selectbox(t(lang_code, "🏥 ICU / OR", "🏥 ICU / OR"), ["ICU", "OR"])
st.sidebar.divider()


# ===============================
# Sidebar: Model settings
# ===============================
st.sidebar.header(t(lang_code, "⚙️ Model Settings", "⚙️ إعدادات النموذج"))
threshold = st.sidebar.slider(t(lang_code, "Threshold (manual)", "العتبة Threshold"), 0.01, 0.99, 0.11)
use_gate = st.sidebar.checkbox(t(lang_code, "Enable Gate", "تفعيل Gate"), value=True)

drop_mode = st.sidebar.selectbox(
    t(lang_code, "Drop Type Mode", "وضع نوع الهبوط"),
    ["AUTO", "A", "B", "C"],
    index=0
)

drop_text_map = {"A": "A: Rapid", "B": "B: Gradual", "C": "C: Intermittent"}
st.sidebar.caption(t(
    lang_code,
    "AUTO = system decides A/B/C from MAP shape.",
    "AUTO = النظام يحدد A/B/C تلقائياً من شكل MAP."
))
st.sidebar.divider()


# ===============================
# Sidebar: Input mode
# ===============================
st.sidebar.header(t(lang_code, "Input Mode", "طريقة الإدخال"))
input_mode = st.sidebar.radio(t(lang_code, "Input Mode", "طريقة الإدخال"), ["CSV Upload", "Manual Entry"], index=0)


# ===============================
# Helpers
# ===============================
def normalize_input_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Required: time, MAP, HR, SpO2
    Optional: RR, EtCO2
    Also enforces numeric types + sorts by time.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required = ["time", "MAP", "HR", "SpO2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if "RR" not in df.columns:
        df["RR"] = np.nan
    if "EtCO2" not in df.columns:
        df["EtCO2"] = np.nan

    # Force numeric
    for c in ["time", "MAP", "HR", "SpO2", "RR", "EtCO2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # IMPORTANT: ensure time is float to avoid merge_asof dtype mismatch
    df["time"] = df["time"].astype(float)

    df = df.sort_values("time").reset_index(drop=True)
    return df


def align_features_to_expected(X: pd.DataFrame, expected_cols_list) -> pd.DataFrame:
    """
    Force EXACT column order/names as trained.
    Missing -> NaN (imputer handles)
    Extra -> dropped
    """
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


def apply_drop_weighting(df_out: pd.DataFrame, scores_df: pd.DataFrame, mode: str):
    """
    Merge drop scores onto df_out by time (nearest).
    Also sets df_out['drop_auto'] when mode == 'AUTO'.
    NOTE: We keep it simple and robust.
    """
    d = df_out.copy()
    s = scores_df.copy()

    # Ensure time types match (float) to avoid MergeError
    d["time"] = pd.to_numeric(d["time"], errors="coerce").astype(float)
    s["time"] = pd.to_numeric(s["time"], errors="coerce").astype(float)

    d = d.sort_values("time").reset_index(drop=True)
    s = s.sort_values("time").reset_index(drop=True)

    # merge_asof nearest
    d = pd.merge_asof(d, s, on="time", direction="nearest")

    # If scores_df already computed drop_auto, keep it; otherwise compute from A/B/C columns
    if "drop_auto" not in d.columns:
        # try to infer from available scores columns
        score_cols = [c for c in ["score_A", "score_B", "score_C"] if c in d.columns]
        if score_cols:
            # pick max score per row
            idx = d[score_cols].to_numpy().argmax(axis=1)
            mapping = {0: "A", 1: "B", 2: "C"}
            d["drop_auto"] = [mapping.get(int(i), "A") for i in idx]
        else:
            d["drop_auto"] = "A"

    # Optional: if you want to weight risk_score by chosen drop type score:
    # We'll do minimal safe logic:
    if mode == "AUTO":
        # choose row-wise key from drop_auto
        if all(col in d.columns for col in ["score_A", "score_B", "score_C"]):
            weights = []
            for k, a, b, c in zip(d["drop_auto"], d["score_A"], d["score_B"], d["score_C"]):
                if k == "B":
                    weights.append(b)
                elif k == "C":
                    weights.append(c)
                else:
                    weights.append(a)
            w = np.array(weights, dtype=float)
            w = np.clip(w, 0.5, 1.5)  # keep stable
            d["risk_score"] = np.clip(d["risk_score"] * w, 0, 1)
    else:
        # manual mode A/B/C
        col = f"score_{mode}"
        if col in d.columns:
            w = np.array(d[col], dtype=float)
            w = np.clip(w, 0.5, 1.5)
            d["risk_score"] = np.clip(d["risk_score"] * w, 0, 1)

    # update alarm after weighting
    d["alarm"] = d["risk_score"].apply(lambda s: generate_alarm(s, threshold))

    return d


def run_inference(df_raw: pd.DataFrame, threshold: float, use_gate: bool, drop_mode: str):
    """
    drop_mode: "AUTO" or "A" or "B" or "C"
    Returns:
      df_out, gate_mask, scores_df, gate_key_used
    """
    df = normalize_input_df(df_raw)

    # 1) Drop scores (A/B/C)
    scores_df = compute_drop_scores(df)

    # 2) Feature extraction
    X = build_feature_matrix(df, expected_cols=expected_cols)

    # 3) Decide which key to use for gating
    # If AUTO: gate initially with "A" (safe), then we will output gate_key_used later.
    gate_key_for_gate = (drop_mode if drop_mode in ["A", "B", "C"] else "A")

    # 4) Gate (optional)
    gate_mask = None
    if use_gate:
        X, gate_mask = safe_apply_gate(X, drop_key=gate_key_for_gate)

    # 5) Align features
    X = align_features_to_expected(X, expected_cols)

    # 6) Predict
    probs = model.predict_proba(X)[:, 1]

    df_out = df.copy()
    df_out["risk_score"] = probs
    df_out["alarm"] = df_out["risk_score"].apply(lambda s: generate_alarm(s, threshold))

    # 7) Drop weighting + drop_auto
    df_out = apply_drop_weighting(df_out, scores_df, mode=drop_mode)

    # 8) FIX KeyError: drop_auto -> read from df_out not scores_df
    if drop_mode == "AUTO":
        if "drop_auto" in df_out.columns:
            vc = df_out["drop_auto"].value_counts()
            gate_key_used = vc.idxmax() if len(vc) else "A"
        else:
            gate_key_used = "A"
    else:
        gate_key_used = drop_mode

    return df_out, gate_mask, scores_df, gate_key_used


def compare_drop_types(df_raw: pd.DataFrame, threshold: float, use_gate: bool):
    rows = []
    for key in ["A", "B", "C"]:
        try:
            df_out, _, _, gate_key_used = run_inference(df_raw, threshold=threshold, use_gate=use_gate, drop_mode=key)
            last = df_out.iloc[-1]
            rows.append({
                "Drop Type": drop_text_map[key],
                "Last MAP": float(last["MAP"]),
                "Last Risk": float(last["risk_score"]),
                "Alarm": "YES 🚨" if bool(last["alarm"]) else "NO ✅",
                "Gate Used": gate_key_used
            })
        except Exception as e:
            rows.append({
                "Drop Type": drop_text_map[key],
                "Last MAP": np.nan,
                "Last Risk": np.nan,
                "Alarm": f"ERROR: {e}",
                "Gate Used": "-"
            })
    return pd.DataFrame(rows)


# ===============================
# Input UI
# ===============================
df_input = None

if input_mode == "CSV Upload":
    uploaded_file = st.file_uploader(t(lang_code, "Upload patient CSV file", "رفع ملف CSV للمريض"), type=["csv"])
    st.info(t(
        lang_code,
        "CSV must contain at least: time, MAP, HR, SpO2 (RR & EtCO2 optional).",
        "يجب أن يحتوي CSV على الأقل: time, MAP, HR, SpO2 (و RR و EtCO2 اختياري)."
    ))
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)

else:
    st.subheader(t(lang_code, "🧾 Manual Entry", "🧾 إدخال يدوي"))
    st.caption(t(
        lang_code,
        "Enter vitals as a time series. Increase points for longer signals.",
        "أدخل الحيويات كسلسلة زمنية. زد عدد النقاط لطول أكبر."
    ))

    n_points = st.number_input(t(lang_code, "Number of time points", "عدد النقاط الزمنية"),
                               min_value=1, max_value=300, value=16, step=1)

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

    rr_start = st.number_input(t(lang_code, "RR start (optional)", "RR بداية (اختياري)"), value=16.0)
    rr_end = st.number_input(t(lang_code, "RR end (optional)", "RR نهاية (اختياري)"), value=28.0)

    etc_start = st.number_input(t(lang_code, "EtCO2 start (optional)", "EtCO2 بداية (اختياري)"), value=35.0)
    etc_end = st.number_input(t(lang_code, "EtCO2 end (optional)", "EtCO2 نهاية (اختياري)"), value=40.0)

    if st.button(t(lang_code, "Generate Manual Timeseries", "توليد سلسلة زمنية يدويًا")):
        t_arr = np.arange(int(n_points), dtype=float) * float(step_time) + float(start_time)
        df_input = pd.DataFrame({
            "time": t_arr,
            "MAP": np.linspace(map_start, map_end, int(n_points)),
            "HR": np.linspace(hr_start, hr_end, int(n_points)),
            "SpO2": np.linspace(spo2_start, spo2_end, int(n_points)),
            "RR": np.linspace(rr_start, rr_end, int(n_points)),
            "EtCO2": np.linspace(etc_start, etc_end, int(n_points)),
        })


if df_input is None:
    st.info(t(lang_code, "⬅️ Choose an input method and provide data.", "⬅️ اختر طريقة إدخال ثم وفّر بيانات."))
    st.stop()


# ===============================
# Run + Display
# ===============================
try:
    df_norm = normalize_input_df(df_input)

    patient_info = {
        "Patient ID": patient_id,
        "Age": age,
        "Sex": sex,
        "ICU/OR": location,
        "Drop Mode": drop_mode
    }

    st.subheader(t(lang_code, "📈 Raw Vitals", "📈 الحيويات الخام"))
    chart_cols = ["HR", "MAP", "SpO2"]
    if "RR" in df_norm.columns:
        chart_cols.append("RR")
    if "EtCO2" in df_norm.columns:
        chart_cols.append("EtCO2")
    st.line_chart(df_norm[chart_cols])

    df_out, gate_mask, scores_df, gate_key_used = run_inference(
        df_norm,
        threshold=threshold,
        use_gate=use_gate,
        drop_mode=drop_mode
    )

    st.subheader(t(lang_code, "🚨 Alarm Timeline", "🚨 خط الإنذار الزمني"))
    st.line_chart(df_out[["risk_score"]])

    latest = df_out.iloc[-1]

    st.subheader(t(lang_code, "🩺 Current Status", "🩺 الحالة الحالية"))
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAP", f"{latest['MAP']:.1f}")
    c2.metric(t(lang_code, "Risk Score", "درجة الخطر"), f"{latest['risk_score']:.3f}")
    c3.metric(t(lang_code, "Alarm", "إنذار"), "YES 🚨" if latest["alarm"] else "NO ✅")
    c4.metric(t(lang_code, "Drop Mode", "وضع الهبوط"), drop_mode)
    c5.metric(t(lang_code, "Auto/Used", "المستخدم/التلقائي"), gate_key_used)

    # Explanation
    st.subheader(t(lang_code, "🧠 Medical Explanation (auto)", "🧠 تفسير طبي (آلي)"))
    exp = build_medical_explanation(
        df_out,
        threshold=threshold,
        drop_key=gate_key_used,
        use_gate=use_gate,
        lang=lang_code
    )

    if bool(latest["alarm"]):
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

    # PDF
    st.subheader(t(lang_code, "📄 PDF Report", "📄 تقرير PDF"))
    pdf_bytes = generate_pdf_report(
        df_out=df_out,
        patient_info=patient_info,
        explanation=exp,
        threshold=threshold,
        drop_text=(drop_text_map.get(gate_key_used, gate_key_used) if drop_mode == "AUTO" else drop_text_map.get(drop_mode, drop_mode)),
        lang=lang_code
    )
    st.download_button(
        t(lang_code, "⬇️ Download PDF Report", "⬇️ تحميل تقرير PDF"),
        data=pdf_bytes,
        file_name=f"{patient_id}_report.pdf",
        mime="application/pdf"
    )

    # Debug / model columns
    with st.expander(t(lang_code, "Show expected model columns", "إظهار أعمدة النموذج المتوقعة")):
        st.write(list(expected_cols))

    with st.expander(t(lang_code, "Show drop scores (head)", "عرض درجات الهبوط (أول صفوف)")):
        st.dataframe(scores_df.head(20), use_container_width=True)

    # Compare A/B/C
    st.subheader(t(lang_code, "🔁 Compare A / B / C (same data)", "🔁 مقارنة A / B / C (نفس البيانات)"))
    comp_df = compare_drop_types(df_norm, threshold=threshold, use_gate=use_gate)
    st.dataframe(comp_df, use_container_width=True)

    # Download CSV output
    st.download_button(
        t(lang_code, "⬇️ Download output CSV (with risk/alarm)", "⬇️ تحميل نتائج CSV (الخطر/الإنذار)"),
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name=f"{patient_id}_output.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error(t(lang_code, "Error during inference:", "خطأ أثناء الاستدلال:"))
    st.exception(e)
