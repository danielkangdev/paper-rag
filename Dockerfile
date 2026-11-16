FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# CPU-only torch first so sentence-transformers doesn't pull CUDA wheels.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY scirag ./scirag
COPY config ./config
RUN pip install ".[models]"

COPY data/corpus.txt ./data/corpus.txt
COPY eval ./eval
COPY scripts ./scripts

# Model weights are NOT baked in: they download on first use into $HF_HOME,
# which docker-compose mounts as a named volume so they persist.
EXPOSE 8000
CMD ["uvicorn", "scirag.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
