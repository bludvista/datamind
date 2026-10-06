import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agents"))

from ingestion_agent import run_ingestion_agent
from eda_agent import run_eda_agent
from visualization_agent import run_visualization_agent
from rag_agent import run_rag_agent  # loads the embedding model once, at import
from insight_agent import run_insight_agent
from report_agent import run_report_agent


class PipelineError(Exception):
    """Raised when a pipeline step fails; keeps the step name and original error."""

    def __init__(self, step: str, cause: Exception):
        super().__init__(f"{step} failed: {cause}")
        self.step = step
        self.cause = cause


def run_pipeline_steps(data_path: str, output_dir: str = "outputs", report_dir: str = "reports") -> dict:
    """Run every agent in order and return all results. Raises PipelineError on failure."""
    results = {}
    steps = [
        ("Agent 1 — Ingestion",      lambda: run_ingestion_agent(data_path)),
        ("Agent 2 — EDA",            lambda: run_eda_agent(results["data"])),
        ("Agent 3 — Visualization",  lambda: run_visualization_agent(results["eda"], output_dir=output_dir)),
        ("Agent 6 — RAG",            lambda: run_rag_agent(results["eda"])),
        ("Agent 4 — Insights",       lambda: run_insight_agent(results["eda"], rag_context=results["rag"]["rag_context"])),
        ("Agent 5 — Report",         lambda: run_report_agent(results["insights"], output_dir=report_dir)),
    ]

    keys = ["data", "eda", "charts", "rag", "insights", "report"]

    for i, (name, step) in enumerate(steps):
        try:
            print(f"\n▶ Running {name}...")
            results[keys[i]] = step()
            print(f"✅ {name} completed!")
        except Exception as e:
            raise PipelineError(name, e) from e

    return results


def run_pipeline(data_path: str):
    try:
        results = run_pipeline_steps(data_path)
    except PipelineError as e:
        print(f"❌ {e.step} FAILED: {e.cause}")
        print("🛑 Pipeline stopped.")
        return None

    return results["report"]


if __name__ == "__main__":
    run_pipeline("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
