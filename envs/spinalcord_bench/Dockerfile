# Hugging Face Spaces / Docker — OpenEnv server only (no bundled GGUF weights).
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md openenv.yaml ./
COPY __init__.py models.py client.py baseline_run.py ./
COPY server ./server

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir .

ENV ENABLE_WEB_INTERFACE=false
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

# HF sets PORT; default to 7860 for local `docker run`.
CMD ["/bin/sh", "-c", "exec uvicorn spinalcord_bench.server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
