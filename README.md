# SpinalCord LLM — Project by Shivansh Darji (AppDice)

**Repository:** [github.com/Shivanshdarji/Spinal-cord](https://github.com/Shivanshdarji/Spinal-cord)

> A revolutionary speculative-decoding based LLM architecture inspired by the biological spinal cord reflex arc.
> A small "Draft" model (Spinal Cord) predicts tokens; a large "Verify" model (Brain) confirms them â€” delivering 2xâ€“3x inference speed.

## Architecture Overview

```
[User Input]
     â”‚
     â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  DRAFT MODEL (CPU)  â”‚  â† "Spinal Cord" â€” Fast, lightweight, instant reflexes
â”‚  SpinalCordDraft    â”‚    Runs on CPU / small VRAM slice
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
          â”‚  Candidate tokens (speculative)
          â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  VERIFY MODEL (GPU) â”‚  â† "Brain" â€” Deep reasoning, final authority
â”‚  SpinalCordBrain    â”‚    Runs on RTX 2050 (CUDA)
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
          â”‚  Accepted / Rejected tokens
          â–¼
     [Output Text]
```

## Project Structure

```
spinalcord/
â”œâ”€â”€ train/              # Python: Design & train the architecture
â”‚   â”œâ”€â”€ model.py        # SpinalCord architecture (PyTorch)
â”‚   â”œâ”€â”€ train.py        # Training loop
â”‚   â”œâ”€â”€ dataset.py      # Data loading utilities
â”‚   â””â”€â”€ config.py       # Hyperparameters
â”œâ”€â”€ convert/            # Convert .pt â†’ .gguf for llama.cpp
â”‚   â””â”€â”€ convert_to_gguf.py
â”œâ”€â”€ inference/          # C++ llama.cpp integration
â”‚   â”œâ”€â”€ spinalcord.cpp  # Main inference engine
â”‚   â””â”€â”€ CMakeLists.txt
â”œâ”€â”€ dashboard/          # Test UI (HTML) + run_dashboard*.bat
â”‚   â”œâ”€â”€ index.html
â”‚   â””â”€â”€ run_dashboard*_prod.bat  # bind 0.0.0.0 for LAN
â”œâ”€â”€ deploy/             # docker-compose (nginx edge) + deploy notes
â”œâ”€â”€ scripts/            # Utilities (e.g. inference quality matrix)
â””â”€â”€ README.md
```

## Setup Guide

### Phase 1: Python Training Environment
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate
```

### Phase 2: C++ Build (llama.cpp)
```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DLLAMA_CUDA=ON
cmake --build build --config Release
```

### Phase 3: Run Dashboard (llama-server + chat UI)
1. Convert checkpoints to GGUF: `python convert/convert_both.py`
2. Start the API + static UI: run `dashboard/run_dashboard.bat` (or `llama-server` with the same flags). The dashboard is served at `http://127.0.0.1:8080`.

**Chat returns HTTP 400 / empty reply:** SpinalCord GGUF uses a **Jinja** chat template. `llama-server` must be started with **`--jinja`** (already included in `run_dashboard.bat`). Without it, `POST /v1/chat/completions` fails.

**HTTP 400 "model name is missing" / "model not found":** Use the dashboard box **â€œFind your model id hereâ€** (Refresh list â†’ **Use for chat** on the **Brain** row). See **`docs/LLAMA_SERVER_MODEL_ID.md`**. You can also open **`http://127.0.0.1:8080/v1/models`** and search for **`"id"`** in the JSON.

**Bad / repetitive / gibberish outputs:** Compare **Brain-only vs Draft+Brain** and **greedy vs sampled** before retraining. Run **`python scripts/diagnose_inference_quality.py`** (PyTorch) and see **`docs/INFERENCE_QUALITY_DEBUG.md`**. For `llama-server`, use **`dashboard/run_dashboard_brain_only.bat`** (no draft) vs **`dashboard/run_dashboard.bat`** (speculative).

**Something that used to work suddenly doesnâ€™t:** See **`docs/TROUBLESHOOTING.md`**. Quick API check: **`powershell -ExecutionPolicy Bypass -File scripts\verify_llama_server.ps1`** (with `llama-server` running).

**Repetitive / junk chat text (API OK):** Use the dashboard **Sampling** row (**Temp / Max tok / Repeat**) â€” values persist in the browser. See **`docs/INFERENCE_QUALITY_DEBUG.md`** Â§0. For **normal Llama-3.2 chat** while SpinalCord trains, run **`dashboard\run_dashboard_llama_scaffold.bat`** (see **`docs/CONVERSATION_TRAINING.md`**).

**Pluggable Brain profiles:** See **`docs/BRAINPACKS.md`** and run `python scripts/run_brainpack.py --pack spinalcord_custom`.

**A/B speed test on llama-server (same model, draft on/off):** run `scripts/benchmark_llama_server.ps1` with `dashboard/run_dashboard_llama_scaffold.bat` vs `dashboard/run_dashboard_llama_scaffold_brain_only.bat`.

## Speed (10x-style) vs quality â€” how it works

- **Speculative decoding**: the Draft proposes `gamma` tokens per round; the Brain verifies in **one** forward pass. Theoretical cap is about **`gamma`Ã—** throughput vs â€œone Brain token per stepâ€.
- **Real speed** depends on **acceptance**: how often Draft matches Brain. After training, **distill the Draft from the Brain** (`train/distill_draft.py`) so acceptance stays high.
- **`generate_reflex()`** (in `train/model.py`): uses Draft+Brain while acceptance is healthy; can fall back to Brain-only for hard spans, then **automatically tries speculative again** after a short Brain-only burst so you donâ€™t lose speed for the whole answer.

Key knobs on `SpinalCordLLM.generate_reflex()`:

| Parameter | Role |
|-----------|------|
| `accept_rate_threshold` | Rolling quality bar (default ~0.65; raise stricter, lower faster) |
| `consecutive_bad_rounds_to_fallback` | Avoid one unlucky round killing Draft (default 2) |
| `recover_speculative_after_brain_tokens` | After N Brain-only tokens, retry Draft+Brain (default 24) |
| `rolling_accept_window` | Smooth acceptance over last N rounds (default 4) |

## â€œAnswer anythingâ€ training (broad prompts)

1. Train Brain on mixed data: `python train/train_brain.py --data_mode mixed`
2. Distill Draft on the same mix: `python train/distill_draft.py --data_mode mixed --brain_ckpt models/scbrain_best.pt`
3. Optional facts: set `SPINALCORD_RAG_DIR` to a folder of `.txt` files when running `test_spinalcord_generate.py` for retrieval-augmented prompts.

## Pluggable SpinalCord (bring your own Brain)

- Core abstraction: `train/pluggable_spinalcord.py` (`SpinalCordEngine`, `BrainAdapter`, `DraftAdapter`).
- Demo runner: `python scripts/demo_pluggable_spinalcord.py` (current checkpoints via adapters).
- BrainPack config: `configs/brainpacks.json` (named profiles).
- BrainPack launcher: `python scripts/run_brainpack.py --pack spinalcord_custom --prompt "hello"`.
- External brain quick test: `python scripts/run_brainpack.py --pack llama_server_chat --prompt "hello"` (uses running llama-server endpoint).
- Goal: keep acceptance/reflex logic fixed while swapping Brain/Draft backends in future.

## Conversation-first training (stories + chat + Q&A)

Bias the model toward **simple language**, **multi-turn dialogue**, and **instruction-style answers**:

1. `python train/train_brain.py --data_mode conversation`
2. `python train/distill_draft.py --data_mode conversation --brain_ckpt models/scbrain_best.pt`

See **`docs/CONVERSATION_TRAINING.md`** for mix weights (`--conv_story`, `--conv_dialog`, `--conv_inst`) and caveats.

Test integrated Draft+Brain + reflex: `python -u test_spinalcord_generate.py`

## OpenEnv — SpinalCord Bench (hackathon)

The **`envs/spinalcord_bench`** package is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible environment: three graded text tasks, `openenv.yaml`, WebSocket client, Dockerfile for **Hugging Face Spaces**, and `openenv validate` (requires `openenv-core` + `uv lock`).

- **Server:** `pip install -e envs/spinalcord_bench` then `uvicorn spinalcord_bench.server.app:app --port 7860`
- **Baseline:** point `OPENAI_BASE_URL` at your **`llama-server`** (same stack as `dashboard/run_dashboard*.bat`) and run `python envs/spinalcord_bench/baseline_run.py`
- **Docs:** see **`envs/spinalcord_bench/README.md`**

## Deploy (LAN / demo)

- **Single machine / LAN:** run `dashboard/run_dashboard_prod.bat` (binds `0.0.0.0` by default). Optional firewall: `deploy/open_firewall_llama.ps1` (Administrator PowerShell). Llama scaffold variant: `dashboard/run_dashboard_llama_scaffold_prod.bat`.
- **Docker edge (nginx + UI, API on host):** see **`deploy/README.md`** and `deploy/docker-compose.yml` (requires Docker Desktop).
- **Cloud GPU VPS:** see **`deploy/CLOUD.md`** and `deploy/docker-compose.cloud.example.yml`.

