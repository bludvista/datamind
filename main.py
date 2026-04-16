import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agents"))

from ingestion_agent import run_ingestion_agent
from eda_agent import run_eda_agent
from visualization_agent import run_visualization_agent
from rag_agent import run_rag_agent
from insight_agent import run_insight_agent
from report_agent import run_report_agent

def main():
    print("=" * 60)
    print("   🧠 DataMind — Agentic AI Pipeline Starting...")
    print("=" * 60)

    print("\n📥 Running Agent 1: Data Ingestion...")
    data = run_ingestion_agent("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

    print("\n🔍 Running Agent 2: Exploratory Data Analysis...")
    eda = run_eda_agent(data)

    print("\n🎨 Running Agent 3: Visualization...")
    run_visualization_agent(eda)

    print("\n🔍 Running Agent 6: RAG Context Retrieval...")
    rag = run_rag_agent(eda)

    print("\n🤖 Running Agent 4: AI Insight Generation...")
    insights = run_insight_agent(eda, rag_context=rag["rag_context"])

    print("\n📝 Running Agent 5: Report Writing...")
    report_path = run_report_agent(insights)

    print("\n" + "=" * 60)
    print("   ✅ DataMind Pipeline Complete!")
    print(f"   📂 Report: {report_path}")
    print("   🌐 Open the report in your browser!")
    print("=" * 60)

if __name__ == "__main__":
    main()