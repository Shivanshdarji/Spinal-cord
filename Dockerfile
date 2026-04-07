# Hugging Face Spaces: build context is the repository root.
# Installs the OpenEnv app from envs/spinalcord_bench (API only; no model weights).
FROM python:3.11-slim

WORKDIR /app

COPY envs/spinalcord_bench/pyproject.toml envs/spinalcord_bench/README.md envs/spinalcord_bench/openenv.yaml ./
COPY envs/spinalcord_bench/__init__.py envs/spinalcord_bench/models.py envs/spinalcord_bench/client.py envs/spinalcord_bench/baseline_run.py ./
COPY envs/spinalcord_bench/server ./server

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir .

ENV ENABLE_WEB_INTERFACE=false
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["/bin/sh", "-c", "exec uvicorn spinalcord_bench.server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
