import streamlit as st
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agents"))

# PAGE CONFIG
st.set_page_config(
    page_title="DataMind — Agentic AI Pipeline",
    page_icon="🧠",
    layout="wide"
)

# HEADER
st.title("🧠 DataMind — Agentic AI Pipeline")
st.markdown("**Automated HR Attrition Analysis powered by Multi-Agent AI + RAG**")
st.divider()

# FILE UPLOAD
uploaded_file = st.file_uploader("📁 Upload your HR CSV file", type=["csv"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.success(f"✅ File uploaded: {uploaded_file.name}")
    st.divider()

    if st.button("▶ Run Analysis", type="primary", use_container_width=True):

        # Import agents ONLY when button is clicked
        from ingestion_agent import run_ingestion_agent
        from eda_agent import run_eda_agent
        from visualization_agent import run_visualization_agent
        from rag_agent import run_rag_agent
        from insight_agent import run_insight_agent
        from report_agent import run_report_agent

        results = {}

        with st.status("🤖 Agent 1 — Data Ingestion...", expanded=False) as status:
            results["data"] = run_ingestion_agent(tmp_path)
            status.update(label="✅ Agent 1 — Data Ingestion Complete!", state="complete")

        with st.status("🤖 Agent 2 — Exploratory Data Analysis...", expanded=False) as status:
            results["eda"] = run_eda_agent(results["data"])
            status.update(label="✅ Agent 2 — EDA Complete!", state="complete")

        st.subheader("📊 Key Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Attrition Rate", f"{results['eda']['attrition_rate']}%")
        col2.metric("Avg Age (Left)", results['eda']['avg_age_left'])
        col3.metric("Avg Age (Stayed)", results['eda']['avg_age_stayed'])
        st.divider()

        with st.status("🤖 Agent 3 — Generating Visualizations...", expanded=False) as status:
            results["charts"] = run_visualization_agent(results["eda"])
            status.update(label="✅ Agent 3 — Charts Ready!", state="complete")

        st.subheader("📈 Visualizations")
        chart_files = [
            "outputs/chart1_attrition_by_dept.png",
            "outputs/chart2_overtime_attrition.png",
            "outputs/chart3_income_distribution.png",
            "outputs/chart4_age_distribution.png"
        ]
        cols = st.columns(2)
        for i, chart_path in enumerate(chart_files):
            if os.path.exists(chart_path):
                cols[i % 2].image(chart_path, use_container_width=True)
        st.divider()

        with st.status("🤖 Agent 6 — RAG Context Retrieval...", expanded=False) as status:
            results["rag"] = run_rag_agent(results["eda"])
            status.update(label="✅ Agent 6 — RAG Complete!", state="complete")

        with st.status("🤖 Agent 4 — AI Insight Generation...", expanded=False) as status:
            results["insights"] = run_insight_agent(
                results["eda"],
                rag_context=results["rag"]["rag_context"]
            )
            status.update(label="✅ Agent 4 — Insights Generated!", state="complete")

        st.subheader("🧠 AI Generated Insights")
        st.markdown(results["insights"]["insights"])
        st.divider()

        with st.status("🤖 Agent 5 — Writing Report...", expanded=False) as status:
            results["report"] = run_report_agent(results["insights"])
            status.update(label="✅ Agent 5 — Report Ready!", state="complete")

        report_path = results["report"] if isinstance(results["report"], str) else None
        if report_path and os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.download_button(
                label="⬇ Download HTML Report",
                data=html_content,
                file_name="hr_attrition_report.html",
                mime="text/html",
                use_container_width=True
            )

        st.success("🎉 Pipeline Complete! All agents finished successfully.")