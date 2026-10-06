import logging
import os

import pytest
from fastapi.testclient import TestClient

import api
from conftest import FAKE_INSIGHTS


@pytest.fixture
def client(workdir, mock_llm, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return TestClient(api.app)


def upload(client, content: bytes, filename: str = "hr.csv"):
    return client.post("/analyze", files={"file": (filename, content, "text/csv")})


@pytest.fixture
def csv_bytes(sample_df):
    return sample_df.to_csv(index=False).encode("utf-8")


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_returns_pipeline_results(client, csv_bytes, workdir, mock_llm):
    response = upload(client, csv_bytes)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"request_id", "ingestion", "eda", "charts", "insights", "report_path"}
    assert body["ingestion"]["total_rows"] == 8
    assert "dataframe" not in body["ingestion"]
    assert "dataframe" not in body["eda"]
    assert body["eda"]["attrition_rate"] == 37.5
    assert body["eda"]["overtime_vs_attrition"] == {"No": 20.0, "Yes": 66.67}
    assert body["eda"]["satisfaction_vs_attrition"] == {"1": 100.0, "2": 50.0, "3": 0.0, "4": 0.0}
    assert len(body["charts"]) == 4
    assert body["insights"] == FAKE_INSIGHTS
    assert os.path.exists(workdir / body["report_path"])
    mock_llm.assert_called_once()


def test_analyze_writes_charts_to_request_folder(client, csv_bytes, workdir):
    body = upload(client, csv_bytes).json()

    request_dir = os.path.join("outputs", body["request_id"])
    assert len(body["charts"]) == 4
    for path in body["charts"]:
        assert os.path.dirname(path) == request_dir
        assert os.path.exists(workdir / path)


def test_analyze_requests_do_not_share_output_folders(client, csv_bytes, workdir):
    first = upload(client, csv_bytes).json()
    first_mtimes = {path: os.path.getmtime(workdir / path) for path in first["charts"]}

    second = upload(client, csv_bytes).json()

    assert first["request_id"] != second["request_id"]
    assert set(first["charts"]).isdisjoint(second["charts"])
    assert sorted(os.listdir(workdir / "outputs")) == sorted([first["request_id"], second["request_id"]])
    # The second request must leave the first request's charts untouched
    assert first_mtimes == {path: os.path.getmtime(workdir / path) for path in first["charts"]}


def test_analyze_requests_get_separate_reports(client, csv_bytes, workdir):
    first = upload(client, csv_bytes).json()
    second = upload(client, csv_bytes).json()

    assert os.path.dirname(first["report_path"]) == os.path.join("outputs", first["request_id"])
    assert os.path.dirname(second["report_path"]) == os.path.join("outputs", second["request_id"])
    assert first["report_path"] != second["report_path"]
    assert os.path.exists(workdir / first["report_path"])
    assert os.path.exists(workdir / second["report_path"])
    assert not (workdir / "reports").exists()


def test_analyze_no_attrition_yes_rows_returns_400(client, sample_df, workdir, mock_llm):
    sample_df["Attrition"] = "No"
    content = sample_df.to_csv(index=False).encode("utf-8")

    response = upload(client, content)

    assert response.status_code == 400
    assert 'no rows with Attrition == "Yes"' in response.json()["detail"]
    assert not (workdir / "outputs").exists()
    mock_llm.assert_not_called()


def test_analyze_empty_file_returns_400(client, mock_llm):
    response = upload(client, b"")

    assert response.status_code == 400
    assert "empty" in response.json()["detail"]
    mock_llm.assert_not_called()


def test_analyze_whitespace_only_file_returns_400(client):
    response = upload(client, b"  \n\n ")

    assert response.status_code == 400


def test_analyze_header_only_csv_returns_400(client):
    response = upload(client, b"Age,Attrition,Department,MonthlyIncome,OverTime,JobSatisfaction\n")

    assert response.status_code == 400
    assert "no data rows" in response.json()["detail"]


def test_analyze_non_csv_bytes_returns_400(client):
    response = upload(client, b"\xff\xfe\x00\x01\x80\x81binary", filename="image.png")

    assert response.status_code == 400
    assert "not a valid CSV" in response.json()["detail"]


def test_analyze_malformed_csv_returns_400(client):
    response = upload(client, b'a,b\n1,2\n3,4,5,6\n')

    assert response.status_code == 400
    assert "not a valid CSV" in response.json()["detail"]


def test_analyze_missing_columns_returns_400(client, sample_df, mock_llm):
    content = sample_df.drop(columns=["Attrition", "OverTime"]).to_csv(index=False).encode("utf-8")

    response = upload(client, content)

    assert response.status_code == 400
    assert "Attrition, OverTime" in response.json()["detail"]
    mock_llm.assert_not_called()


def test_analyze_without_file_returns_422(client):
    assert client.post("/analyze").status_code == 422


def test_analyze_missing_api_key_returns_clear_error(client, csv_bytes, mock_llm, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")

    response = upload(client, csv_bytes)

    assert response.status_code == 503
    assert "OPENROUTER_API_KEY is not set" in response.json()["detail"]
    mock_llm.assert_not_called()


def test_analyze_pipeline_failure_returns_500_and_logs(client, csv_bytes, mock_llm, caplog):
    mock_llm.side_effect = RuntimeError("LLM down")

    with caplog.at_level(logging.ERROR, logger="datamind.api"):
        response = upload(client, csv_bytes)

    assert response.status_code == 500
    assert "Agent 4 — Insights" in response.json()["detail"]
    assert "LLM down" in response.json()["detail"]
    assert any("Pipeline failed" in record.message for record in caplog.records)
    assert any(record.exc_info for record in caplog.records)
