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
The same pipeline is available as a REST API (`api.py`).

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

### Error codes
Errors are returned as `{"detail": "<message>"}`.

| Status | When |
|---|---|
| 400 | The file is empty, is not a valid CSV, has no data rows, is missing a required column, or has no rows with `Attrition` equal to `Yes` |
| 422 | No `file` field was sent |
| 500 | A pipeline step failed; the message names the step and the traceback is logged |
| 503 | `OPENROUTER_API_KEY` is not set on the server |

## Running the Tests
```
pytest
```

The tests mock the OpenRouter client and the embedding model, so they run without
an API key or internet access.

## Known Limitations
- The per-request folders under `outputs/` are not cleaned up automatically.
- The FAISS index over `rag_docs/` is rebuilt on every request; only the embedding model is loaded once and reused.
