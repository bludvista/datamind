import os

import pytest

from conftest import FAKE_INSIGHTS
from ingestion_agent import run_ingestion_agent
from eda_agent import run_eda_agent
from visualization_agent import run_visualization_agent
from insight_agent import run_insight_agent
from report_agent import run_report_agent
import rag_agent


@pytest.fixture
def eda_results(sample_csv):
    return run_eda_agent(run_ingestion_agent(sample_csv))


# Agent 1 — Ingestion

def test_ingestion_summarizes_csv(sample_csv):
    result = run_ingestion_agent(sample_csv)

    assert result["total_rows"] == 8
    assert result["total_columns"] == 6
    assert result["missing_values"] == 0
    assert sorted(result["numerical_columns"]) == ["Age", "JobSatisfaction", "MonthlyIncome"]
    assert sorted(result["categorical_columns"]) == ["Attrition", "Department", "OverTime"]
    assert len(result["dataframe"]) == 8


def test_ingestion_counts_missing_values(sample_df, tmp_path):
    sample_df.loc[0, "MonthlyIncome"] = None
    sample_df.loc[1, "Department"] = None
    path = tmp_path / "missing.csv"
    sample_df.to_csv(path, index=False)

    assert run_ingestion_agent(str(path))["missing_values"] == 2


def test_ingestion_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_ingestion_agent(str(tmp_path / "nope.csv"))


# Agent 2 — EDA

def test_eda_computes_attrition_stats(eda_results):
    assert eda_results["attrition_rate"] == 37.5
    assert eda_results["attrition_by_department"] == {
        "Human Resources": 50.0,
        "Research & Development": 25.0,
        "Sales": 50.0,
    }
    assert eda_results["avg_income_by_department"] == {
        "Human Resources": 3000.0,
        "Research & Development": 5875.0,
        "Sales": 4000.0,
    }
    assert eda_results["overtime_vs_attrition"] == {"No": 20.0, "Yes": 66.67}
    assert eda_results["satisfaction_vs_attrition"] == {1: 100.0, 2: 50.0, 3: 0.0, 4: 0.0}
    assert eda_results["avg_age_left"] == 28.0
    assert eda_results["avg_age_stayed"] == 41.6


def test_eda_missing_column_raises(sample_df):
    with pytest.raises(KeyError):
        run_eda_agent({"dataframe": sample_df.drop(columns=["Attrition"])})


# Agent 3 — Visualization

def test_visualization_saves_four_charts(eda_results, tmp_path):
    output_dir = tmp_path / "charts"

    paths = run_visualization_agent(eda_results, output_dir=str(output_dir))

    assert [os.path.basename(p) for p in paths] == [
        "chart1_attrition_by_dept.png",
        "chart2_overtime_attrition.png",
        "chart3_income_distribution.png",
        "chart4_age_distribution.png",
    ]
    for path in paths:
        assert os.path.getsize(path) > 0


# Agent 4 — Insights

def test_insight_agent_returns_llm_output(eda_results, mock_llm):
    result = run_insight_agent(eda_results, rag_context="Overtime drives burnout.")

    assert result["insights"] == FAKE_INSIGHTS
    assert result["eda_results"] is eda_results
    mock_llm.assert_called_once()


def test_insight_agent_prompt_includes_stats_and_context(eda_results, mock_llm):
    run_insight_agent(eda_results, rag_context="Overtime drives burnout.")

    prompt = mock_llm.call_args.kwargs["messages"][0]["content"]
    assert "37.5%" in prompt
    assert "28.0" in prompt
    assert "Overtime drives burnout." in prompt


def test_insight_agent_without_rag_context(eda_results, mock_llm):
    run_insight_agent(eda_results)

    prompt = mock_llm.call_args.kwargs["messages"][0]["content"]
    assert "No additional context available." in prompt


# Agent 5 — Report

def test_report_agent_writes_html(eda_results, workdir):
    path = run_report_agent({"insights": FAKE_INSIGHTS, "eda_results": eda_results})

    assert os.path.dirname(path) == "reports" and path.endswith(".html")
    html = (workdir / path).read_text(encoding="utf-8")
    assert "37.5%" in html
    assert "<td>Sales</td><td>50.0%</td>" in html
    assert FAKE_INSIGHTS in html


def test_report_agent_writes_to_output_dir(eda_results, workdir):
    output_dir = os.path.join("outputs", "abc")

    path = run_report_agent({"insights": FAKE_INSIGHTS, "eda_results": eda_results}, output_dir=output_dir)

    assert os.path.dirname(path) == output_dir
    assert os.path.exists(workdir / path)
    assert not (workdir / "reports").exists()


# Agent 6 — RAG

def write_docs(folder):
    folder.mkdir()
    (folder / "overtime.txt").write_text("Overtime overtime overtime leads to burnout.", encoding="utf-8")
    (folder / "pay.txt").write_text("Salary salary benchmarks matter.", encoding="utf-8")
    (folder / "notes.md").write_text("ignored", encoding="utf-8")


def test_load_documents_reads_only_txt(tmp_path):
    write_docs(tmp_path / "docs")

    documents = rag_agent.load_documents(str(tmp_path / "docs"))

    assert sorted(doc["filename"] for doc in documents) == ["overtime.txt", "pay.txt"]


def test_load_documents_creates_missing_folder(tmp_path):
    folder = tmp_path / "docs"

    assert rag_agent.load_documents(str(folder)) == []
    assert folder.is_dir()


def test_retrieve_returns_closest_document_first(tmp_path):
    write_docs(tmp_path / "docs")
    documents = rag_agent.load_documents(str(tmp_path / "docs"))
    index, contents = rag_agent.build_vector_store(documents)

    context = rag_agent.retrieve_relevant_context("overtime overtime overtime", index, contents, top_k=1)

    assert context == "Overtime overtime overtime leads to burnout."


def test_rag_agent_returns_context(eda_results, workdir):
    write_docs(workdir / "rag_docs")

    result = rag_agent.run_rag_agent(eda_results)

    assert "Overtime overtime overtime leads to burnout." in result["rag_context"]
    assert "Salary salary benchmarks matter." in result["rag_context"]
    assert result["eda_results"] is eda_results


def test_rag_agent_without_documents(eda_results, workdir):
    result = rag_agent.run_rag_agent(eda_results)

    assert result["rag_context"] == ""
