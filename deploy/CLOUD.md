# Deploy on the cloud

Yes. The usual pattern is a **Linux GPU VM** running **`llama-server`** in Docker (or natively), with your **GGUF** files and the **`dashboard/`** folder on disk.

## What you need

| Piece | Notes |
|--------|------|
| **GPU** | NVIDIA T4 / L4 / A10 or better for 3B+ at usable speed. CPU-only is possible but slow. |
| **OS** | Ubuntu 22.04 LTS is the most common. |
| **Docker** | Docker Engine + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). |
| **Models** | Upload `models/*.gguf` to the server (scp, S3, Hugging Face CLI, etc.). |
| **Flags** | Custom SpinalCord GGUF needs **`--jinja`** (same as local). |

## Quick path: Compose on a GPU VPS

1. Rent a **GPU instance** (examples: [RunPod](https://www.runpod.io/), [Lambda Labs](https://lambdalabs.com/), AWS **g4dn**, GCP **G2**, Azure **NC** series).
2. SSH in, install Docker + NVIDIA container toolkit.
3. Clone this repo (or copy `dashboard/` + `models/` + `deploy/`).
4. Copy `deploy/docker-compose.cloud.example.yml` → `deploy/docker-compose.cloud.yml` and adjust **`--model` / `--model-draft`** paths to match your filenames.
5. From `deploy/`:

```bash
docker compose -f docker-compose.cloud.yml pull
docker compose -f docker-compose.cloud.yml up -d
```

6. Open **`http://<public-ip>:8080/`** (or bind to localhost and SSH tunnel).

If the upstream image tag moves, set `image:` to a specific tag from [ghcr.io/ggml-org/llama.cpp](https://github.com/orgs/ggml-org/packages/container/package/llama.cpp).

## Security (do not skip)

- **Do not** leave `0.0.0.0:8080` on a public IP without **TLS**, **auth**, and **rate limits**. Anyone can run inference on your bill.
- Minimum hardening: SSH tunnel only (`ssh -L 8080:127.0.0.1:8080 user@server`), or a VPN, or a reverse proxy with **HTTPS** + **API key** middleware.
- For production, put **Caddy** or **nginx** + **Let’s Encrypt** in front and avoid exposing `llama-server` directly.

## Brain-only (no draft) in the cloud

Omit `--model-draft` and draft-related flags; use a single `--model` line. Easiest when VRAM is tight.

## Managed “LLM APIs”

Fully managed providers (OpenAI, Together, etc.) host the model for you; they do **not** run your local `llama-server` binary. You can still call them from your own apps, but that is **not** this repo’s default stack.

## Cost tip

Stop the instance or `docker compose down` when not benchmarking; GPU hours add up quickly.

## “Free” cloud — what is realistic

There is **no general-purpose provider that gives a persistent, public GPU VM for free** that is suitable for running **3B+ `llama-server`** the same way you do on a desktop. Free tiers are almost always **CPU-only**, **very small RAM**, **sleep after idle**, or **notebook sessions** (not a long-lived server).

| Option | Reality |
|--------|--------|
| **Google Colab / Kaggle (free GPU)** | Good for **experiments**, not a stable **deploy**. Sessions end, queues exist, and the workflow is **notebooks**, not “Docker + port 8080” unless you work around it. |
| **Oracle Cloud “Always Free”** | **ARM** VMs (no NVIDIA GPU on the free shape). You *might* run **tiny** GGUF on **CPU** with `llama-server` for a demo; expect **slow** responses and tight RAM. |
| **Fly.io / Railway / Render (free tier)** | Often **sleeps**, low CPU/RAM; large GGUF + `llama-server` usually **does not fit** or hits cold-start limits. |
| **Hugging Face Spaces** | Can host **demo UIs**; it is a **different** packaging model (Gradio/Streamlit) than this repo’s **dashboard + llama-server** stack unless you re-wrap it. |
| **“Free” tunnel (Cloudflare Tunnel, ngrok)** | **Free** way to share a **URL** — but **inference still runs on your PC**. The cloud is only the tunnel, not the GPU. |

**Practical “free” demos for SpinalCord today**

1. **Run locally** (`run_dashboard*.bat`) and share with **`cloudflared tunnel`** or **ngrok** (no GPU in the cloud; your machine does the work).
2. Use **Colab/Kaggle** only if you accept **notebook-style** runs and **no guaranteed uptime**.
3. For **real** cloud GPU deploy, budget for **pay-as-you-go** (often **$0.20–$1+/hr** class hardware) or use **credits** (GCP/AWS student/startup programs) — still not “free forever,” but can be **free for a limited trial**.

If a product advertises “free unlimited LLM hosting,” read the **VRAM, rate limits, and sleep policy**; it rarely matches a self-hosted **llama-server + your GGUF** setup.
