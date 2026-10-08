# DataMind

[![CI](https://github.com/bludvista/datamind/actions/workflows/ci.yml/badge.svg)](https://github.com/bludvista/datamind/actions/workflows/ci.yml)

DataMind is a multi-agent pipeline that analyses HR attrition data. You upload a CSV, and a chain of specialised agents profiles the data, runs exploratory analysis, draws charts, retrieves domain knowledge with RAG, asks an LLM for insights, and writes a downloadable HTML report.

It runs two ways: as a **Streamlit app** for interactive use, and as an **async FastAPI service** with a streaming endpoint, a Docker image and CI.

## How it works

The orchestrator runs six agents in order:

| Step | Agent | What it does |
|------|-------|--------------|
| 1 | Ingestion | Loads and validates the CSV, profiles rows, columns and missing values |
| 2 | EDA | Computes attrition rate and breakdowns by department, overtime, satisfaction, age and income |
| 3 | Visualization | Saves four charts (attrition by department, overtime vs attrition, income and age distributions) |
| 4 | RAG | Retrieves relevant HR-domain passages from `rag_docs/` (Sentence Transformers `all-MiniLM-L6-v2` + FAISS) |
| 5 | Insight | Sends the EDA results and retrieved context to an LLM via OpenRouter and gets key findings, root causes and recommendations |
| 6 | Report | Builds a self-contained HTML report |

The LLM insights can be streamed token by token. The CPU-bound steps run in worker threads (`asyncio.to_thread`), so the API event loop stays responsive.

## Tech stack

Python 3.12, FastAPI, Uvicorn, Streamlit, Pandas, Matplotlib, Sentence Transformers, FAISS, OpenRouter (AsyncOpenAI client), pytest, Docker, GitHub Actions.

## Quick start

```bash
git clone https://github.com/bludvista/datamind.git
cd datamind
python -m venv venv
venv\Scripts\activate          # Windows PowerShell; on Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=your-key-here
```

Never commit `.env`; it is listed in `.gitignore`.

### Dataset

The pipeline is built for the IBM HR Analytics Employee Attrition dataset (`WA_Fn-UseC_-HR-Employee-Attrition.csv`, 1,470 rows, 35 columns), available on Kaggle. The dataset is not included in this repository. Any CSV that has the required columns works: `Attrition`, `Department`, `MonthlyIncome`, `OverTime`, `JobSatisfaction`, `Age`.

### Run the Streamlit app

```bash
streamlit run app.py
```

### Run the API

```bash
uvicorn api:app --reload
```

The first start takes up to a minute because the embedding model is loaded. Wait for `Application startup complete.`, then open http://localhost:8000/docs for interactive documentation.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check, returns `{"status": "ok"}` |
| POST | `/analyze` | Upload a CSV (multipart field `file`); returns one JSON result with `request_id`, `ingestion`, `eda`, `charts`, `insights`, `report_path` |
| POST | `/analyze/stream` | Same input; streams the result as server-sent events |

### Streaming example

```bash
curl.exe -N -X POST -F "file=@data/WA_Fn-UseC_-HR-Employee-Attrition.csv" http://localhost:8000/analyze/stream
```

(On Linux/macOS use `curl` instead of `curl.exe`.) The `-N` flag turns off buffering so you see tokens as they arrive.

Event sequence:

```
event: metadata   data: {"request_id", "ingestion", "eda", "charts"}
event: token      data: {"text": "<chunk>"}          (repeated)
event: done       data: {"insights", "report_path"}
event: error      data: {"detail": "..."}            (only if a step fails mid-stream)
```

### Errors

| Status | Meaning |
|--------|---------|
| 400 | Invalid upload: empty file, not a CSV, no data rows, missing required columns, or no `Attrition == "Yes"` rows |
| 422 | The `file` field is missing |
| 500 | A pipeline step failed (the message names the step) |
| 503 | `OPENROUTER_API_KEY` is not set |

Failures that happen after a stream has started are sent as an `error` event, because the HTTP status has already been sent.

## Docker

```bash
docker build -t datamind .
docker run --rm -p 8000:8000 -e OPENROUTER_API_KEY=your-key-here -v "$(pwd)/outputs:/app/outputs" datamind
```

The image is based on `python:3.12-slim` and uses CPU-only PyTorch to keep it small. The key is passed at run time and never baked into the image.

## Tests and CI

```bash
pytest
```

The tests mock the LLM and the embedding model, so they need no API key and no network. They cover the agents, the orchestrator and the API (including error mapping and the SSE stream).

GitHub Actions runs on every push and pull request:

1. **Tests**: Python 3.12, CPU-only PyTorch, `pytest`.
2. **Docker image**: builds the image, starts the container and polls `/health` until it answers.

## Project structure

```
agents/          one module per agent (ingestion, EDA, visualization, RAG, insight, report)
rag_docs/        domain documents used for retrieval
tests/           pytest suite with mocked LLM and embeddings
api.py           FastAPI service
app.py           Streamlit app
orchestrator.py  runs the agents in order (sync and async variants)
main.py          command-line entry point
Dockerfile
.github/workflows/ci.yml
```

## Known limitations

- Generated charts and reports in `outputs/` are not cleaned up automatically.
- The FAISS index is rebuilt for every request.
- CPU-bound steps run in a thread pool, which limits throughput under heavy concurrent load.
- Insights come from an LLM and describe correlations in the data, not proven causes. Review them before acting on them.
- There is no authentication on the API yet.

## Author

Shashank Kumar Sadashiva, [github.com/bludvista](https://github.com/bludvista)
