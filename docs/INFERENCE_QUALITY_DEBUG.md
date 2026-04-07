# Inference quality debugging (SpinalCord)

If chat or local generation looks like gibberish, endless repetition, or nonsense, **narrow the failure** before changing training:

1. **Brain weights vs Draft/spec pipeline**
2. **Sampling vs greedy**
3. **PyTorch (training tokenizer) vs GGUF + `llama-server`**

---

## 0) Dashboard quick knobs (llama-server chat)

On **`http://127.0.0.1:8080`**, use the **Sampling** row above the chat box:

| Control | Default | Purpose |
|--------|---------|--------|
| **Temp** | 0.2 | Lower → more greedy, less random junk on small models |
| **Max tok** | 256 | Shorter replies = less room for repetition loops |
| **Repeat** | 1.15 | `repeat_penalty` (llama-server JSON); reduces `sococo…` style loops |

Values are **saved in the browser**. For training-limited models, prefer **temp ≤ 0.3** and **repeat ≥ 1.1**.

---

## 1) PyTorch matrix (recommended first)

From the repo root, with `models/scbrain_best.pt` and `models/scdraft_best.pt` present:

```bash
python scripts/diagnose_inference_quality.py
```

Optional:

```bash
python scripts/diagnose_inference_quality.py --prompt "Your failing prompt here" --max-new-tokens 128
python scripts/diagnose_inference_quality.py --modes brain,spec --greedy-only
python scripts/diagnose_inference_quality.py --modes reflex --sampled-only --sampled-temp 0.7
```

**How to read the output**

| Observation | Likely cause |
|-------------|----------------|
| **Brain-only** is bad (gibberish / repeat) | Brain checkpoint, tokenizer/vocab mismatch, or need more/better training data |
| Brain-only OK, **spec/reflex** bad | Draft distillation/speculative path; try server **without draft** (below) or re-distill |
| **Greedy** OK, **sampled** wild | Normal for small models; lower temperature / top-p in UI or server |

`generate_brain_only()` uses **true argmax** when `temperature <= 0`. Speculative paths use the same sampling helpers as the rest of the stack (near-greedy when `temperature=0`).

---

## 2) `llama-server`: Brain-only vs Draft+Brain

**Windows:** If `run_dashboard_brain_only.bat` fails with `. was unexpected at this time`, common causes are: **(1)** unescaped `(` `)` inside an `echo` line—escape as `^(` `^)`; **(2)** `set PATH=%PATH%;...` when `%PATH%` contains `Program Files (x86)`—use `set "PATH=%PATH%;..."` so parentheses stay inside the quoted value.

**Draft + Brain (default dashboard):** `dashboard/run_dashboard.bat`  
Uses `--model` (brain GGUF) + `--model-draft` (draft GGUF) + `--jinja`.

**Brain only (no speculative draft):** `dashboard/run_dashboard_brain_only.bat`  
Same brain GGUF, **no** `--model-draft`. Chat should match “plain” autoregressive behavior on the brain export.

For Llama scaffold A/B:

- Draft + brain: `dashboard/run_dashboard_llama_scaffold.bat`
- Brain-only: `dashboard/run_dashboard_llama_scaffold_brain_only.bat`

Then benchmark the **currently running** server:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\benchmark_llama_server.ps1 -Runs 8 -MaxTokens 128
```

Run that once in each mode and compare:

- `Wall tok/s (mean)`  (end-to-end)
- `Predicted tok/s mean` (server decode metric)

Or run fully automated A/B (starts both modes itself):

```powershell
python scripts/benchmark_serverpack_ab.py `
  --brain models/Llama-3.2-3B-Instruct-Q4_K_M.gguf `
  --draft models/Llama-3.2-1B-Instruct-Q4_K_M.gguf `
  --runs 8 --warmup 3 --max-tokens 128 --target-speedup 2.0
```

This prints:

- Baseline (brain-only) metrics
- SpinalCord-integrated (brain+draft) metrics
- Speedup and PASS/BELOW_TARGET vs your threshold

To auto-search configs toward 2x on your machine:

```powershell
python scripts/tune_to_2x.py `
  --brain models/Llama-3.2-3B-Instruct-Q4_K_M.gguf `
  --draft models/Llama-3.2-1B-Instruct-Q4_K_M.gguf `
  --port 8090 --runs 4 --warmup 1 --target-speedup 2.0
```

`tune_to_2x.py` sweeps:

- `max_tokens`: 128/256/512
- `ngld`: 0 or 99 (draft on CPU vs GPU)
- `draft-max`: 4/6/8
- `draft-min`: 1/2

and ranks the best measured speedups.

Use the dashboard **model id** panel (or `GET http://127.0.0.1:8080/v1/models`) so the client sends the correct `model` name. See `docs/LLAMA_SERVER_MODEL_ID.md`.

**Rough parity with greedy PyTorch:** set temperature to **0** (or minimum) in the chat UI if your build exposes it.

---

## 3) Quick API check (optional)

With the server running:

```bash
curl -s http://127.0.0.1:8080/v1/models
```

Then `POST /v1/chat/completions` with a JSON body that includes `"model": "<id-from-list>"`, `"messages": [...]`, and optionally `"temperature": 0`.

---

## 4) If PyTorch is good but GGUF is bad

Suspect **conversion** (`convert/convert_both.py`), **quantization**, or a **tokenizer/chat-template** mismatch. Re-export and compare a **single** prompt byte-for-byte where possible.

---

## 5) Brain early exit (PyTorch — not in stock GGUF / llama-server)

The Brain can use a **shallow path**: after **K** transformer blocks an auxiliary `early_lm_head` predicts the next distribution; at inference, if **max softmax prob** on the last position is above a threshold, generation **skips** the remaining blocks for that token.

- **Train:** `train/train_brain.py` adds auxiliary CE when `BrainConfig.early_exit_loss_weight > 0` (default `0.25`). CLI: `--early-exit-after 4`, `--early-exit-loss-weight 0.25`.
- **Generate:** `SpinalCordLLM.generate_brain_early_exit(..., early_exit_max_prob=0.88)` in `train/model.py`.
- **Smoke test:** `python scripts/demo_early_exit.py`

Until you **retrain** with the auxiliary loss, the early head is random and will almost always take the **full** depth. This path does **not** export to standard `llama-server` GGUF without custom inference code.

## Files

| File | Role |
|------|------|
| `scripts/diagnose_inference_quality.py` | Brain vs spec vs reflex × greedy vs sampled |
| `train/model.py` → `generate_brain_only()` | Isolated brain decoding |
| `train/model.py` → `generate_brain_early_exit()` | Brain + shallow exit when confident |
| `dashboard/run_dashboard.bat` | Draft + brain server |
| `dashboard/run_dashboard_brain_only.bat` | Brain-only server |
