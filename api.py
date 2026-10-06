import io
import logging
import math
import os
import tempfile
import uuid

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

from orchestrator import PipelineError, run_pipeline_steps

logger = logging.getLogger("datamind.api")

# Columns the EDA and visualization agents read from the uploaded CSV
REQUIRED_COLUMNS = ["Attrition", "Department", "MonthlyIncome", "OverTime", "JobSatisfaction", "Age"]
OUTPUT_ROOT = "outputs"

app = FastAPI(title="DataMind — Agentic AI Pipeline")


def _to_jsonable(value):
    """Convert pipeline output (numpy scalars, non-string keys, NaN) into plain JSON types."""
    if isinstance(value, dict):
        return {str(_to_jsonable(k)): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _validate_csv(content: bytes) -> None:
    if not content or not content.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Uploaded file contains no CSV data.")
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Uploaded file is not a valid CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV has a header but no data rows.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing required columns: {', '.join(missing)}"
        )

    if not (df["Attrition"] == "Yes").any():
        raise HTTPException(
            status_code=400,
            detail='CSV has no rows with Attrition == "Yes", so attrition cannot be analyzed.'
        )


def _build_response(request_id: str, results: dict) -> dict:
    ingestion = {k: v for k, v in results["data"].items() if k != "dataframe"}
    eda = {k: v for k, v in results["eda"].items() if k != "dataframe"}
    return _to_jsonable({
        "request_id": request_id,
        "ingestion": ingestion,
        "eda": eda,
        "charts": results["charts"],
        "insights": results["insights"]["insights"],
        "report_path": results["report"],
    })


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(file: UploadFile = File(...)):
    content = file.file.read()
    _validate_csv(content)

    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY is not set")
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY is not set. Add it to the .env file or the environment and restart the server."
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    # Each request writes its charts and report to its own folder so concurrent requests don't collide
    request_id = uuid.uuid4().hex
    output_dir = os.path.join(OUTPUT_ROOT, request_id)

    try:
        results = run_pipeline_steps(tmp_path, output_dir=output_dir, report_dir=output_dir)
        return _build_response(request_id, results)
    except PipelineError as e:
        logger.exception("Pipeline failed at %s for request %s (file %r)", e.step, request_id, file.filename)
        raise HTTPException(status_code=500, detail=f"Pipeline failed at {e.step}: {e.cause}")
    except Exception:
        logger.exception("Unexpected error for request %s (file %r)", request_id, file.filename)
        raise HTTPException(status_code=500, detail="Internal error while running the pipeline.")
    finally:
        os.remove(tmp_path)
