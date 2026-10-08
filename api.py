import asyncio
import io
import json
import logging
import math
import os
import tempfile
import uuid
from contextlib import contextmanager

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

# orchestrator puts agents/ on sys.path, so it has to be imported before the agents
from orchestrator import PipelineError, run_analysis_steps_async, run_pipeline_steps_async
from insight_agent import stream_insight_agent
from report_agent import run_report_agent

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


def _write_temp_csv(content: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(content)
        return tmp.name


async def _receive_csv(file: UploadFile) -> str:
    """Validate the upload and the server config, then save the CSV to a temp file and return its path."""
    content = await file.read()
    # Parsing the CSV is CPU-bound, so it runs in a worker thread instead of on the event loop
    await asyncio.to_thread(_validate_csv, content)

    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY is not set")
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY is not set. Add it to the .env file or the environment and restart the server."
        )

    return await asyncio.to_thread(_write_temp_csv, content)


@contextmanager
def _pipeline_errors(request_id: str, filename: str):
    """Log pipeline failures and turn them into 500 responses."""
    try:
        yield
    except PipelineError as e:
        logger.exception("Pipeline failed at %s for request %s (file %r)", e.step, request_id, filename)
        raise HTTPException(status_code=500, detail=f"Pipeline failed at {e.step}: {e.cause}")
    except Exception:
        logger.exception("Unexpected error for request %s (file %r)", request_id, filename)
        raise HTTPException(status_code=500, detail="Internal error while running the pipeline.")


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event. The payload is JSON so newlines in the text can't break the framing."""
    return f"event: {event}\ndata: {json.dumps(_to_jsonable(data))}\n\n"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    tmp_path = await _receive_csv(file)

    # Each request writes its charts and report to its own folder so concurrent requests don't collide
    request_id = uuid.uuid4().hex
    output_dir = os.path.join(OUTPUT_ROOT, request_id)

    try:
        with _pipeline_errors(request_id, file.filename):
            results = await run_pipeline_steps_async(tmp_path, output_dir=output_dir, report_dir=output_dir)
            return _build_response(request_id, results)
    finally:
        os.remove(tmp_path)


@app.post("/analyze/stream")
async def analyze_stream(file: UploadFile = File(...)):
    tmp_path = await _receive_csv(file)

    request_id = uuid.uuid4().hex
    output_dir = os.path.join(OUTPUT_ROOT, request_id)

    # Everything before the LLM call runs before the stream opens, so a failure
    # here can still be reported with a real HTTP status code.
    try:
        with _pipeline_errors(request_id, file.filename):
            results = await run_analysis_steps_async(tmp_path, output_dir=output_dir)
    finally:
        os.remove(tmp_path)

    filename = file.filename

    async def event_stream():
        yield _sse("metadata", {
            "request_id": request_id,
            "ingestion": {k: v for k, v in results["data"].items() if k != "dataframe"},
            "eda": {k: v for k, v in results["eda"].items() if k != "dataframe"},
            "charts": results["charts"],
        })

        # The 200 status is already sent by now, so failures are reported as an `error` event
        step = "Agent 4 — Insights"
        try:
            pieces = []
            async for text in stream_insight_agent(results["eda"], rag_context=results["rag"]["rag_context"]):
                pieces.append(text)
                yield _sse("token", {"text": text})

            insights = "".join(pieces)
            step = "Agent 5 — Report"
            report_path = await asyncio.to_thread(
                run_report_agent, {"insights": insights, "eda_results": results["eda"]}, output_dir=output_dir)
        except Exception as e:
            logger.exception("Pipeline failed at %s for request %s (file %r)", step, request_id, filename)
            yield _sse("error", {"detail": f"Pipeline failed at {step}: {e}"})
            return

        yield _sse("done", {"insights": insights, "report_path": report_path})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        # Stop proxies (e.g. nginx) from caching or buffering the events
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
