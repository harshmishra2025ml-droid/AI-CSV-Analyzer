import io, os, requests, pandas as pd, streamlit as st
import plotly.express as px
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="AI CSV Analyzer", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container{max-width:1450px;padding-top:2rem;padding-bottom:4rem}
.hero{padding:1.5rem 1.7rem;border:1px solid rgba(255,255,255,.10);border-radius:20px;
background:linear-gradient(135deg,rgba(99,102,241,.22),rgba(14,165,233,.08));margin-bottom:1.2rem}
.hero h1{margin:0;font-size:2.35rem}.hero p{margin:.45rem 0 0;opacity:.7}
.card{padding:1.05rem 1.2rem;border:1px solid rgba(255,255,255,.10);border-radius:16px;
background:rgba(255,255,255,.035);min-height:105px}
.label{font-size:.78rem;opacity:.62;text-transform:uppercase;letter-spacing:.06em}
.value{font-size:1.8rem;font-weight:700;margin-top:.25rem}
.section{font-size:1.25rem;font-weight:700;margin:1.2rem 0 .6rem}
.insight{padding:.8rem 1rem;border-left:3px solid #6366f1;background:rgba(99,102,241,.07);
border-radius:8px;margin:.45rem 0}
.good{border-left-color:#22c55e}.warn{border-left-color:#f59e0b}.bad{border-left-color:#ef4444}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>📊 AI CSV Analyzer</h1><p>Profile your data, find quality issues, detect anomalies, and get AI-powered explanations.</p></div>', unsafe_allow_html=True)

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

with st.sidebar:
    st.markdown("## ⚙️ Workspace")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.divider()
    st.caption("Backend")
    st.code(BACKEND)
    st.caption("Tip: keep FastAPI running in a second terminal.")

if uploaded and st.button("🚀 Analyze dataset", type="primary", use_container_width=True):
    with st.spinner("Analyzing your dataset..."):
        try:
            r = requests.post(
                f"{BACKEND}/api/analyze",
                files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
                timeout=180,
            )
            r.raise_for_status()
            st.session_state.analysis = r.json()
            st.session_state.csv_bytes = uploaded.getvalue()
            st.session_state.filename = uploaded.name
            st.session_state.ai = None
        except Exception as e:
            st.error(f"Backend error: {e}")

data = st.session_state.get("analysis")
if not data:
    st.info("👈 Upload a CSV from the sidebar to get started.")
    st.markdown("#### What you'll get")
    st.markdown("**Overview** · **Data quality** · **Columns** · **Correlations** · **ML** · **AI Scientist**")
    st.stop()

p, ml = data["profile"], data["ml"]
df = pd.read_csv(io.BytesIO(st.session_state.csv_bytes))

kpis = [
    ("Rows", f"{p['rows']:,}"),
    ("Columns", f"{p['columns_count']:,}"),
    ("Quality", f"{p['quality_score']}/100"),
    ("Missing cells", f"{p['missing_total']:,}"),
]
cols = st.columns(4)
for c, (label, value) in zip(cols, kpis):
    c.markdown(
        f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )

st.write("")
tabs = st.tabs(["🏠 Overview", "🧹 Data Quality", "📋 Columns", "🔗 Correlations", "🧠 Anomally Detection", "✨ AI analysis"])

with tabs[0]:
    left, right = st.columns([1.4, 1])
    with left:
        st.markdown('<div class="section">Dataset preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(15), use_container_width=True, hide_index=True)
    with right:
        nums = df.select_dtypes("number").columns.tolist()
        if nums:
            col = st.selectbox("Distribution", nums)
            fig = px.histogram(df, x=col, marginal="box", height=330)
            fig.update_layout(margin=dict(l=10, r=10, t=15, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns detected.")

    st.markdown('<div class="section">Quick signals</div>', unsafe_allow_html=True)
    for c in sorted(p["columns"], key=lambda x: x["missing_pct"], reverse=True)[:5]:
        if c["missing_pct"] > 0:
            st.markdown(
                f'<div class="insight warn"><b>{c["name"]}</b> · {c["missing_pct"]}% missing · {c["unique"]:,} unique</div>',
                unsafe_allow_html=True,
            )

with tabs[1]:
    a, b = st.columns(2)
    a.metric("Duplicate rows", f"{p['duplicates']:,}")
    a.metric("Memory footprint", f"{p['memory_mb']} MB")
    b.progress(p["quality_score"] / 100, text=f"Quality score: {p['quality_score']}/100")
    b.caption("Heuristic triage score, not a formal quality standard.")

    miss = pd.DataFrame([
        {"Column": c["name"], "Missing %": c["missing_pct"], "Missing": c["missing"], "Unique": c["unique"]}
        for c in p["columns"]
    ])
    st.markdown('<div class="section">Missing-value profile</div>', unsafe_allow_html=True)
    st.dataframe(miss.sort_values("Missing %", ascending=False).head(20),
                 use_container_width=True, hide_index=True)

with tabs[2]:
    rows = [{k: c.get(k) for k in ["name","type","dtype","missing","missing_pct","unique","unique_pct","outliers_iqr"]}
            for c in p["columns"]]
    prof = pd.DataFrame(rows)
    st.dataframe(prof, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download profile CSV", prof.to_csv(index=False),
                       "column_profile.csv", "text/csv")

with tabs[3]:
    corr = pd.DataFrame(p["correlations"])
    if corr.empty:
        st.info("Not enough numeric columns for correlations.")
    else:
        st.dataframe(corr, use_container_width=True, hide_index=True)
        st.caption("Sorted by absolute correlation. Correlation does not imply causation.")

with tabs[4]:
    a = ml.get("anomaly_detection")
    c = ml.get("clustering")
    x, y = st.columns(2)
    with x:
        if a:
            st.metric("Estimated anomalies", f"{a['anomaly_pct']}%")
            st.caption(f"{a['anomalies']} anomalies in a sample of {a['sample_size']:,} complete numeric rows.")
        else:
            st.info("Not enough complete numeric data.")
    with y:
        if c:
            st.metric("Clusters", c["clusters"])
            st.write("Cluster sizes:", c["sizes"])
        else:
            st.info("Clustering needs enough complete numeric data.")

with tabs[5]:
    st.markdown("### ✨ AI insights")
    st.caption("AI receives the structured profile and ML results, not the entire raw CSV.")

    if st.button("Generate AI insights", type="primary"):
        with st.spinner("AI is reviewing the dataset..."):
            try:
                r = requests.post(
                    f"{BACKEND}/api/ai",
                    files={"file": (st.session_state.filename, st.session_state.csv_bytes, "text/csv")},
                    timeout=180,
                )
                r.raise_for_status()
                st.session_state.ai = r.json()
            except Exception as e:
                st.error(f"AI error: {e}")

    ai = st.session_state.get("ai")
    if ai:
        if not ai.get("enabled"):
            st.warning(ai.get("summary", "AI is not configured."))
        else:
            st.success("AI analysis generated")
            st.markdown("#### Executive summary")
            st.write(ai.get("summary", ""))

            sections = [
                ("🔎 Key findings", "key_findings", "good"),
                ("⚠️ Data-quality risks", "data_quality_risks", "warn"),
                ("💡 Recommendations", "recommendations", "good"),
                ("🧠 Modeling notes", "modeling_notes", "bad"),
            ]
            for title, key, kind in sections:
                vals = ai.get(key, [])
                if vals:
                    with st.expander(title, expanded=True):
                        for value in vals:
                            st.markdown(
                                f'<div class="insight {kind}">{value}</div>',
                                unsafe_allow_html=True,
                            )
