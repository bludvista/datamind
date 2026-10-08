import json
import logging
import os

import pytest
from fastapi.testclient import TestClient

import api
from conftest import FAKE_INSIGHTS, FAKE_STREAM_PIECES, fake_llm_stream


@pytest.fixture
def client(workdir, mock_async_llm, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return TestClient(api.app)


def upload(client, content: bytes, filename: str = "hr.csv"):
    return client.post("/analyze", files={"file": (filename, content, "text/csv")})


def upload_stream(client, content: bytes, filename: str = "hr.csv"):
    return client.post("/analyze/stream", files={"file": (filename, content, "text/csv")})


def parse_sse(body: str) -> list:
    """Split an SSE response body into (event name, decoded JSON payload) pairs."""
    events = []
    for block in body.strip().split("\n\n"):
        event_line, data_line = block.split("\n")
        assert event_line.startswith("event: ")
        assert data_line.startswith("data: ")
        events.append((event_line[len("event: "):], json.loads(data_line[len("data: "):])))
    return events


@pytest.fixture
def csv_bytes(sample_df):
    return sample_df.to_csv(index=False).encode("utf-8")


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_returns_pipeline_results(client, csv_bytes, workdir, mock_async_llm):
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
    mock_async_llm.assert_called_once()


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


def test_analyze_no_attrition_yes_rows_returns_400(client, sample_df, workdir, mock_async_llm):
    sample_df["Attrition"] = "No"
    content = sample_df.to_csv(index=False).encode("utf-8")

    response = upload(client, content)

    assert response.status_code == 400
    assert 'no rows with Attrition == "Yes"' in response.json()["detail"]
    assert not (workdir / "outputs").exists()
    mock_async_llm.assert_not_called()


def test_analyze_empty_file_returns_400(client, mock_async_llm):
    response = upload(client, b"")

    assert response.status_code == 400
    assert "empty" in response.json()["detail"]
    mock_async_llm.assert_not_called()


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


def test_analyze_missing_columns_returns_400(client, sample_df, mock_async_llm):
    content = sample_df.drop(columns=["Attrition", "OverTime"]).to_csv(index=False).encode("utf-8")

    response = upload(client, content)

    assert response.status_code == 400
    assert "Attrition, OverTime" in response.json()["detail"]
    mock_async_llm.assert_not_called()


def test_analyze_without_file_returns_422(client):
    assert client.post("/analyze").status_code == 422


def test_analyze_missing_api_key_returns_clear_error(client, csv_bytes, mock_async_llm, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")

    response = upload(client, csv_bytes)

    assert response.status_code == 503
    assert "OPENROUTER_API_KEY is not set" in response.json()["detail"]
    mock_async_llm.assert_not_called()


def test_analyze_pipeline_failure_returns_500_and_logs(client, csv_bytes, mock_async_llm, caplog):
    mock_async_llm.side_effect = RuntimeError("LLM down")

    with caplog.at_level(logging.ERROR, logger="datamind.api"):
        response = upload(client, csv_bytes)

    assert response.status_code == 500
    assert "Agent 4 — Insights" in response.json()["detail"]
    assert "LLM down" in response.json()["detail"]
    assert any("Pipeline failed" in record.message for record in caplog.records)
    assert any(record.exc_info for record in caplog.records)


def test_analyze_uses_async_llm_client(client, csv_bytes, mock_llm, mock_async_llm):
    upload(client, csv_bytes)

    mock_async_llm.assert_awaited_once()
    mock_llm.assert_not_called()


def test_stream_returns_server_sent_events(client, csv_bytes):
    response = upload_stream(client, csv_bytes)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"


def test_stream_event_order(client, csv_bytes):
    events = parse_sse(upload_stream(client, csv_bytes).text)

    names = [name for name, _ in events]
    assert names == ["metadata"] + ["token"] * len(FAKE_STREAM_PIECES) + ["done"]


def test_stream_sends_llm_output_piece_by_piece(client, csv_bytes, mock_async_llm):
    events = parse_sse(upload_stream(client, csv_bytes).text)

    # One event per LLM chunk, in order, with newlines inside the text preserved
    assert [data["text"] for name, data in events if name == "token"] == FAKE_STREAM_PIECES
    mock_async_llm.assert_awaited_once()
    assert mock_async_llm.call_args.kwargs["stream"] is True


def test_stream_metadata_matches_analyze(client, csv_bytes, workdir):
    name, metadata = parse_sse(upload_stream(client, csv_bytes).text)[0]
    analyze_body = upload(client, csv_bytes).json()

    assert name == "metadata"
    assert set(metadata) == {"request_id", "ingestion", "eda", "charts"}
    assert metadata["ingestion"] == analyze_body["ingestion"]
    assert metadata["eda"] == analyze_body["eda"]
    assert len(metadata["charts"]) == 4
    for path in metadata["charts"]:
        assert os.path.dirname(path) == os.path.join("outputs", metadata["request_id"])
        assert os.path.exists(workdir / path)


def test_stream_done_event_has_full_insights_and_report(client, csv_bytes, workdir):
    events = parse_sse(upload_stream(client, csv_bytes).text)
    request_id = events[0][1]["request_id"]
    name, done = events[-1]

    assert name == "done"
    assert done["insights"] == FAKE_INSIGHTS
    assert os.path.dirname(done["report_path"]) == os.path.join("outputs", request_id)
    report = (workdir / done["report_path"]).read_text(encoding="utf-8")
    assert FAKE_INSIGHTS in report


def test_stream_passes_rag_context_to_llm(client, csv_bytes, workdir, mock_async_llm):
    (workdir / "rag_docs").mkdir()
    (workdir / "rag_docs" / "overtime.txt").write_text("Overtime leads to burnout.", encoding="utf-8")

    upload_stream(client, csv_bytes)

    prompt = mock_async_llm.call_args.kwargs["messages"][0]["content"]
    assert "Overtime leads to burnout." in prompt


def test_stream_llm_failure_mid_stream_sends_error_event(client, csv_bytes, workdir, mock_async_llm, caplog):
    async def failing_create(**kwargs):
        return fake_llm_stream(FAKE_STREAM_PIECES[:1], error=RuntimeError("LLM down"))

    mock_async_llm.side_effect = failing_create

    with caplog.at_level(logging.ERROR, logger="datamind.api"):
        response = upload_stream(client, csv_bytes)

    # The status line was sent before the LLM failed, so the failure arrives as an event
    assert response.status_code == 200
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["metadata", "token", "error"]
    assert events[1][1]["text"] == FAKE_STREAM_PIECES[0]
    assert "Agent 4 — Insights" in events[-1][1]["detail"]
    assert "LLM down" in events[-1][1]["detail"]
    assert any(record.exc_info for record in caplog.records)
    # No report is written from a partial answer
    request_dir = workdir / "outputs" / events[0][1]["request_id"]
    assert not [f for f in os.listdir(request_dir) if f.endswith(".html")]


def test_stream_llm_failure_before_first_token_sends_error_event(client, csv_bytes, mock_async_llm):
    mock_async_llm.side_effect = RuntimeError("LLM down")

    events = parse_sse(upload_stream(client, csv_bytes).text)

    assert [name for name, _ in events] == ["metadata", "error"]
    assert "LLM down" in events[-1][1]["detail"]


def test_stream_invalid_csv_returns_400_before_streaming(client, sample_df, workdir, mock_async_llm):
    content = sample_df.drop(columns=["Attrition"]).to_csv(index=False).encode("utf-8")

    response = upload_stream(client, content)

    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"
    assert "Attrition" in response.json()["detail"]
    assert not (workdir / "outputs").exists()
    mock_async_llm.assert_not_called()


def test_stream_empty_file_returns_400(client):
    assert upload_stream(client, b"").status_code == 400


def test_stream_without_file_returns_422(client):
    assert client.post("/analyze/stream").status_code == 422


def test_stream_missing_api_key_returns_503(client, csv_bytes, mock_async_llm, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")

    response = upload_stream(client, csv_bytes)

    assert response.status_code == 503
    assert "OPENROUTER_API_KEY is not set" in response.json()["detail"]
    mock_async_llm.assert_not_called()


def test_stream_failure_before_llm_returns_500(client, csv_bytes, mock_async_llm, monkeypatch):
    import orchestrator

    def broken_rag(eda_results):
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(orchestrator, "run_rag_agent", broken_rag)

    response = upload_stream(client, csv_bytes)

    assert response.status_code == 500
    assert "Agent 6 — RAG" in response.json()["detail"]
    assert "index unavailable" in response.json()["detail"]
    mock_async_llm.assert_not_called()
