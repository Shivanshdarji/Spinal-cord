# Deploy (test / demo)

Two practical options: **all-in-one `llama-server`** (simplest) or **Docker nginx edge** (same-origin UI + API when `llama-server` stays on the host).

## Option A — Single process (recommended)

1. Put GGUF files under `spinalcord/models/` as usual.
2. Open the Windows firewall port you use (default `8080`) for **Inbound TCP** if other devices need access. Administrator PowerShell: `powershell -ExecutionPolicy Bypass -File deploy\open_firewall_llama.ps1 -Port 8080`
3. From the repo root:

```powershell
cd C:\Users\SHIVANSH\Desktop\spinalcord
$env:PORT = "8080"
$env:HOST = "0.0.0.0"
$env:NGLD = "0"
.\dashboard\run_dashboard_prod.bat
```

**LAN deploy with Meta Llama 3.2 + speculative (draft), not your custom `scbrain`/`scdraft`:**  
If both custom GGUFs exist, `run_dashboard_prod.bat` would normally pick them. Force Llama + SpinalCord-style speculative decoding:

```powershell
$env:USE_LLAMA_SCAFFOLD = "1"
.\dashboard\run_dashboard_prod.bat
```

Or run the dedicated script (same stack):

```powershell
.\dashboard\run_dashboard_llama_scaffold_prod.bat
```

4. On another PC or phone (same Wi‑Fi), open **`http://<server-pc-lan-ip>:8080/`** (find the IP with `ipconfig` on Windows).

**Server device + test device:** The PC running `llama-server` is the **server**. Any other device only needs a **browser** — no GGUF, no CUDA. The dashboard detects when you did **not** open it via `localhost` and points **Single chat** and **A/B compare** URLs at **that same host** instead of stale `127.0.0.1` (so the phone does not accidentally call itself).

For **A/B** you still need **two ports** if you want baseline vs speculative: run a second `llama-server` on **8081** on the **same** PC, and **bind both to the LAN** (`HOST=0.0.0.0`). The default brain-only scripts used **`127.0.0.1` only**, so **phones get “failed to fetch”** on the baseline column while single chat (port 8080) still works.

**Example (two PowerShell windows on the server PC):**

```powershell
# Speculative (8080) — already using LAN
$env:PORT='8080'; $env:HOST='0.0.0.0'; $env:NGLD='0'
.\dashboard\run_dashboard_llama_scaffold_prod.bat
```

```powershell
# Baseline brain-only (8081) — must listen on 0.0.0.0 for other devices
$env:PORT='8081'; $env:HOST='0.0.0.0'
.\dashboard\run_dashboard_llama_scaffold_brain_only.bat
```

Open firewall **TCP 8080 and 8081** (run `deploy\open_firewall_llama.ps1` twice with `-Port 8080` and `-Port 8081`, or one rule per port). Compare URLs on the phone should be `http://<lan-ip>:8081` and `http://<lan-ip>:8080` (the UI rewrites `127.0.0.1` to your current host).

Security: this exposes the model API to your LAN. Do not port-forward to the public internet without TLS, auth, and rate limits.

## Option B — Docker nginx in front of host llama-server

Use this when you want the static UI served by nginx while `llama-server` still runs natively with CUDA on Windows.

1. Start `llama-server` on the host bound to **all interfaces**, e.g. `HOST=0.0.0.0` and `PORT=8080` (see Option A).
2. Install Docker Desktop (Windows) or Docker Engine (Linux).
3. From `spinalcord/deploy/`:

```powershell
docker compose up -d
```

4. Open `http://localhost:8088/` (default `EDGE_PORT`). The UI uses relative `/v1/...` calls; nginx forwards them to `host.docker.internal:8080`.

Override ports / upstream:

```powershell
$env:EDGE_PORT = "80"
$env:LLAMA_PORT = "8080"
docker compose up -d
```

**A/B compare mode** still expects two backends (e.g. `8081` baseline and `8080` speculative). With only one `llama-server`, point both compare URLs at the same edge URL or run a second server on another port.

## Cloud (GPU VPS)

See **`CLOUD.md`** and **`docker-compose.cloud.example.yml`** for a Linux + NVIDIA + Docker example (`ghcr.io/ggml-org/llama.cpp:server-cuda`).

## Checklist

- [ ] `--jinja` for custom SpinalCord GGUF (already in the batch files).
- [ ] Low VRAM: set `NGLD=0` (draft CPU) before blaming speed.
- [ ] Health: `GET /v1/models` should return JSON when the server is up.
