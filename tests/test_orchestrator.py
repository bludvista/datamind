import os

import pytest

import orchestrator
from conftest import FAKE_INSIGHTS


def test_run_pipeline_steps_returns_all_results(sample_csv, workdir, mock_llm):
    results = orchestrator.run_pipeline_steps(sample_csv)

    assert list(results) == ["data", "eda", "charts", "rag", "insights", "report"]
    assert results["eda"]["attrition_rate"] == 37.5
    assert len(results["charts"]) == 4
    assert results["insights"]["insights"] == FAKE_INSIGHTS
    assert os.path.exists(results["report"])


def test_run_pipeline_steps_passes_rag_context_to_llm(sample_csv, workdir, mock_llm):
    (workdir / "rag_docs").mkdir()
    (workdir / "rag_docs" / "overtime.txt").write_text("Overtime leads to burnout.", encoding="utf-8")

    results = orchestrator.run_pipeline_steps(sample_csv)

    assert results["rag"]["rag_context"] == "Overtime leads to burnout."
    prompt = mock_llm.call_args.kwargs["messages"][0]["content"]
    assert "Overtime leads to burnout." in prompt


def test_run_pipeline_steps_writes_charts_to_output_dir(sample_csv, workdir, mock_llm):
    results = orchestrator.run_pipeline_steps(sample_csv, output_dir=os.path.join("outputs", "abc"))

    assert len(results["charts"]) == 4
    for path in results["charts"]:
        assert os.path.dirname(path) == os.path.join("outputs", "abc")
        assert os.path.exists(path)


def test_run_pipeline_returns_report_path(sample_csv, workdir, mock_llm):
    report_path = orchestrator.run_pipeline(sample_csv)

    assert os.path.exists(report_path)


def test_run_pipeline_steps_raises_with_failed_step(sample_csv, workdir, mock_llm):
    mock_llm.side_effect = RuntimeError("LLM down")

    with pytest.raises(orchestrator.PipelineError) as excinfo:
        orchestrator.run_pipeline_steps(sample_csv)

    assert excinfo.value.step == "Agent 4 — Insights"
    assert "LLM down" in str(excinfo.value)


def test_run_pipeline_returns_none_on_failure(workdir, mock_llm):
    assert orchestrator.run_pipeline(str(workdir / "missing.csv")) is None
    mock_llm.assert_not_called()
