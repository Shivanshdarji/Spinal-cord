# SpinalCord Bench (OpenEnv)

Hackathon-ready OpenEnv package: **three synthetic “real-world” text tasks** (invoice total, calendar overlap, log error code) with **deterministic graders** in `[0, 1]`, **trajectory shaping** (partial credit, empty/loop penalties), and an **OpenAI-compatible baseline** intended for **SpinalCord** served through `llama-server` (same API as the project dashboard).

## Tasks

| ID | Difficulty | What is graded |
|----|------------|----------------|
| `extract_total` | easy | Model states **47.23** (USD invoice total). |
| `calendar_overlap` | medium | Model says **YES** to overlap with a short reason. |
| `log_first_error` | hard | Model returns **`E42`** (first ERROR line in a tiny log). |

## Run the server (local)

```bash
cd envs/spinalcord_bench
pip install -e .
export ENABLE_WEB_INTERFACE=false
uvicorn spinalcord_bench.server.app:app --host 0.0.0.0 --port 7860
```

Validate:

```bash
openenv validate .
```

## Baseline (SpinalCord / OpenAI-compatible)

1. Start your model: e.g. `dashboard/run_dashboard.bat` (or your `llama-server` command) and note the **model id** from `GET /v1/models`.
2. In another shell:

```bash
set OPENENV_BASE_URL=http://127.0.0.1:7860
set OPENAI_BASE_URL=http://127.0.0.1:8080/v1
set OPENAI_MODEL=<your-model-id>
python baseline_run.py
```

Use `OPENAI_API_KEY` or `HF_TOKEN` if your endpoint requires a key.

## Hugging Face Spaces

- **Recommended:** link the **GitHub repo** and use the **`Dockerfile` at the repository root** (build context = whole repo; copies `envs/spinalcord_bench` into the image).
- Alternative: build only from this folder using `envs/spinalcord_bench/Dockerfile` if your host lets you set the build context to this directory.
- Exposes **7860**; HF sets **`PORT`** automatically.
- Set Space secrets if your app calls a remote API; the **environment server** itself does not download model weights.

Tag the GitHub repository with **`openenv`** for discoverability.

### Hugging Face Space README header (optional)

Paste at the top of the Space `README.md`:

```yaml
---
title: SpinalCord Bench (OpenEnv)
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
---
```

## Wire protocol

- **WebSocket** session at `/ws` (use `SpinalBenchClient`) for multi-step episodes.
- HTTP `/reset` and `/step` are stateless in `openenv-core`; prefer the client for persistent state.
