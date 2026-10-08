# 🧠 DataMind — Agentic AI Pipeline

Automated HR Attrition Analysis powered by Multi-Agent AI, RAG & LLM.

## Features
- 6 autonomous AI agents working in sequence
- RAG system using HuggingFace Sentence Transformers + FAISS
- LLM-generated HR insight reports via OpenRouter API
- Interactive Streamlit UI with CSV upload
- Auto-generated charts and downloadable HTML report

## Tech Stack
Python | Streamlit | OpenRouter API | RAG | HuggingFace | FAISS | Pandas | Matplotlib

## How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your `OPENROUTER_API_KEY` to `.env`
4. Run: `streamlit run app.py`

## FastAPI Service
The same pipeline is available as an async REST API (`api.py`), with a streaming
variant that sends the LLM insights as they are generated.

Run it from the project root:

```
uvicorn api:app --reload
```

The server starts on `http://127.0.0.1:8000`, with interactive docs at `/docs`.
`OPENROUTER_API_KEY` must be set in `.env` or the environment.

### GET /health
```
curl http://127.0.0.1:8000/health
```
```json
{"status": "ok"}
```

### POST /analyze
Upload a CSV as multipart form data in the `file` field. The CSV must contain the
columns `Attrition`, `Department`, `MonthlyIncome`, `OverTime`, `JobSatisfaction`
and `Age`, and at least one row with `Attrition` equal to `Yes`.

```
curl -X POST http://127.0.0.1:8000/analyze -F "file=@data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
```

Example response (abbreviated):

```json
{
  "request_id": "3f2b8c1e9a7d4e5f8b6a1c2d3e4f5a6b",
  "ingestion": {
    "total_rows": 1470,
    "total_columns": 35,
    "missing_values": 0,
    "numerical_columns": ["Age", "DailyRate", "..."],
    "categorical_columns": ["Attrition", "BusinessTravel", "..."]
  },
  "eda": {
    "attrition_rate": 16.12,
    "attrition_by_department": {"Human Resources": 19.05, "Research & Development": 13.84, "Sales": 20.63},
    "avg_income_by_department": {"Human Resources": 6654.51, "...": 0},
    "overtime_vs_attrition": {"No": 10.44, "Yes": 30.53},
    "satisfaction_vs_attrition": {"1": 22.84, "...": 0},
    "avg_age_left": 33.61,
    "avg_age_stayed": 37.56
  },
  "charts": [
    "outputs/3f2b8c1e9a7d4e5f8b6a1c2d3e4f5a6b/chart1_attrition_by_dept.png",
    "outputs/3f2b8c1e9a7d4e5f8b6a1c2d3e4f5a6b/chart2_overtime_attrition.png",
    "outputs/3f2b8c1e9a7d4e5f8b6a1c2d3e4f5a6b/chart3_income_distribution.png",
    "outputs/3f2b8c1e9a7d4e5f8b6a1c2d3e4f5a6b/chart4_age_distribution.png"
  ],
  "insights": "KEY FINDINGS\n- ...",
  "report_path": "outputs/3f2b8c1e9a7d4e5f8b6a1c2d3e4f5a6b/hr_attrition_report_20261006_101500.html"
}
```

Each request writes its charts and HTML report to its own `outputs/<request_id>/`
folder. Paths are relative to the directory the server was started from, and use
backslashes on Windows.

### POST /analyze/stream
Takes the same upload as `/analyze`, but streams the insight agent's LLM output as
[Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
while it is being generated, instead of waiting for the full answer.

```
curl -N -X POST http://127.0.0.1:8000/analyze/stream -F "file=@data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
```

`-N` turns off curl's output buffering so the events appear as they arrive.

```
event: metadata
data: {"request_id": "3f2b...", "ingestion": {...}, "eda": {...}, "charts": ["outputs/3f2b.../chart1_attrition_by_dept.png", "..."]}

event: token
data: {"text": "KEY FINDINGS\n- Overall attrition is "}

event: token
data: {"text": "16.12%, with Sales the highest"}

event: done
data: {"insights": "KEY FINDINGS\n- Overall attrition is 16.12%, ...", "report_path": "outputs/3f2b.../hr_attrition_report_20261006_101500.html"}
```

| Event | Sent | Data |
|---|---|---|
| `metadata` | Once, first | `request_id`, `ingestion`, `eda` and `charts`, the same values `/analyze` returns |
| `token` | Once per piece of LLM output | `text`: the next piece of the insights. Concatenate them in order |
| `done` | Once, last, on success | `insights` (the full text) and `report_path` |
| `error` | Once, last, if the LLM call or the report step fails | `detail`: the failed step and the error |

Every `data:` line is one JSON object. A stream ends with either `done` or `error`.

Ingestion, EDA, charts and RAG run before the stream opens, so problems with the
upload or with those steps return the normal status codes below. Once the stream
has started the status is already `200`, so a later failure is reported as an
`error` event and no report is written.

The endpoint is a POST with a file upload, so the browser `EventSource` API (GET
only) cannot call it. Use `fetch` and read the response body as a stream.

### Async design
The route handlers are `async`. The LLM call uses `AsyncOpenAI`, so a request
waiting on OpenRouter does not hold a thread. The CPU-bound steps (CSV parsing,
pandas, matplotlib, embeddings) run in worker threads via `asyncio.to_thread`, so
they do not block the event loop either. `orchestrator.run_pipeline_steps` and
`insight_agent.run_insight_agent` are still synchronous and are what the Streamlit
app and `main.py` use; the API uses `run_pipeline_steps_async` and
`stream_insight_agent`.

### Error codes
Errors are returned as `{"detail": "<message>"}`.

| Status | When |
|---|---|
| 400 | The file is empty, is not a valid CSV, has no data rows, is missing a required column, or has no rows with `Attrition` equal to `Yes` |
| 422 | No `file` field was sent |
| 500 | A pipeline step failed; the message names the step and the traceback is logged |
| 503 | `OPENROUTER_API_KEY` is not set on the server |

## Docker
The FastAPI service can run in a container. Build the image from the project root:

```
docker build -t datamind .
```

Run it, passing the API key from the environment. The key is not stored in the
image, and `.env` is excluded from the build by `.dockerignore`.

```
docker run --rm -p 8000:8000 -e OPENROUTER_API_KEY=your-key datamind
```

Or reuse your existing `.env` file (lines must be plain `KEY=value`, without quotes):

```
docker run --rm -p 8000:8000 --env-file .env datamind
```

The API is then available at `http://127.0.0.1:8000`, exactly as with `uvicorn`:

```
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/analyze -F "file=@data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
```

Charts and reports are written to `/app/outputs` inside the container and are lost
when it is removed. Mount a volume to keep them:

```
docker run --rm -p 8000:8000 --env-file .env -v datamind-outputs:/app/outputs datamind
```

About the image:
- It is based on `python:3.12-slim`, runs as a non-root user and serves on port 8000.
- PyTorch is installed as the CPU-only build, to keep the image smaller.
- The embedding model is downloaded while the image is built, so the first build needs internet access and the container starts without contacting Hugging Face.
- It contains the API only (`api.py`, `orchestrator.py`, `agents/`, `rag_docs/`), not the Streamlit app. Rebuild the image after changing the documents in `rag_docs/`.

## Running the Tests
```
pytest
```

The tests mock the OpenRouter client and the embedding model, so they run without
an API key or internet access.

## Continuous Integration
`.github/workflows/ci.yml` runs on every push and pull request. It installs the
dependencies and runs `pytest`, then builds the Docker image, starts a container
and checks that `GET /health` responds. No secrets are needed.

## Known Limitations
- The per-request folders under `outputs/` are not cleaned up automatically.
- Async helps with waiting, not with computation: the pandas, matplotlib and embedding steps still take CPU time in worker threads, so heavy parallel load needs more uvicorn workers.
- The FAISS index over `rag_docs/` is rebuilt on every request; only the embedding model is loaded once and reused.
