import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agents"))

from ingestion_agent import run_ingestion_agent
from eda_agent import run_eda_agent
from visualization_agent import run_visualization_agent
from rag_agent import run_rag_agent  # loads the embedding model once, at import
from insight_agent import run_insight_agent, run_insight_agent_async
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


async def _run_step_async(name: str, awaitable):
    try:
        print(f"\n▶ Running {name}...")
        result = await awaitable
        print(f"✅ {name} completed!")
        return result
    except Exception as e:
        raise PipelineError(name, e) from e


async def run_analysis_steps_async(data_path: str, output_dir: str = "outputs") -> dict:
    """Run the steps that come before the LLM call (ingestion, EDA, charts, RAG).

    These are CPU-bound (pandas, matplotlib, embeddings), so each one runs in a
    worker thread to keep the event loop free. Raises PipelineError on failure.
    """
    results = {}
    results["data"] = await _run_step_async(
        "Agent 1 — Ingestion", asyncio.to_thread(run_ingestion_agent, data_path))
    results["eda"] = await _run_step_async(
        "Agent 2 — EDA", asyncio.to_thread(run_eda_agent, results["data"]))
    results["charts"] = await _run_step_async(
        "Agent 3 — Visualization", asyncio.to_thread(run_visualization_agent, results["eda"], output_dir=output_dir))
    results["rag"] = await _run_step_async(
        "Agent 6 — RAG", asyncio.to_thread(run_rag_agent, results["eda"]))
    return results


async def run_pipeline_steps_async(data_path: str, output_dir: str = "outputs", report_dir: str = "reports") -> dict:
    """Async version of run_pipeline_steps: same steps, order and results, without blocking the event loop."""
    results = await run_analysis_steps_async(data_path, output_dir=output_dir)
    results["insights"] = await _run_step_async(
        "Agent 4 — Insights", run_insight_agent_async(results["eda"], rag_context=results["rag"]["rag_context"]))
    results["report"] = await _run_step_async(
        "Agent 5 — Report", asyncio.to_thread(run_report_agent, results["insights"], output_dir=report_dir))
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
