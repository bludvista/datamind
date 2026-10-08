FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib

# Run as a non-root user; /app/outputs is where each request's charts and report are written
RUN useradd --create-home appuser \
    && mkdir -p /app/outputs \
    && chown -R appuser:appuser /app
WORKDIR /app

# Dependencies come before the source code so this layer is reused when only code changes.
# PyTorch is installed from the CPU-only index first: the default Linux wheels bundle
# CUDA and add several GB that this service never uses.
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

USER appuser

# rag_agent loads this embedding model at import time. Downloading it during the build
# means the container starts without needing to reach the Hugging Face Hub.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1

COPY --chown=appuser:appuser api.py orchestrator.py ./
COPY --chown=appuser:appuser agents/ agents/
COPY --chown=appuser:appuser rag_docs/ rag_docs/

# OPENROUTER_API_KEY is not set here on purpose: pass it at run time (docker run -e / --env-file)
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
