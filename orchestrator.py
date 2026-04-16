import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agents"))

from ingestion_agent import run_ingestion_agent
from eda_agent import run_eda_agent
from visualization_agent import run_visualization_agent
from insight_agent import run_insight_agent
from report_agent import run_report_agent

def run_pipeline(data_path: str):
    results = {}
    steps = [
        ("Agent 1 — Ingestion",      lambda: run_ingestion_agent(data_path)),
        ("Agent 2 — EDA",            lambda: run_eda_agent(results["data"])),
        ("Agent 3 — Visualization",  lambda: run_visualization_agent(results["eda"])),
        ("Agent 4 — Insights",       lambda: run_insight_agent(results["eda"])),
        ("Agent 5 — Report",         lambda: run_report_agent(results["insights"])),
    ]

    keys = ["data", "eda", "charts", "insights", "report"]

    for i, (name, step) in enumerate(steps):
        try:
            print(f"\n▶ Running {name}...")
            results[keys[i]] = step()
            print(f"✅ {name} completed!")
        except Exception as e:
            print(f"❌ {name} FAILED: {e}")
            print("🛑 Pipeline stopped.")
            return None

    return results["report"]


if __name__ == "__main__":
    run_pipeline("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")