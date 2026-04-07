#!/usr/bin/env python3
"""
Automatic A/B benchmark for llama-server:
  A) Brain-only (baseline)
  B) Brain + Draft (SpinalCord-integrated speculative path)

You can swap Brain/Draft GGUF paths directly from CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from typing import Any
from urllib import request, error


def _http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 120) -> dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = request.Request(url=url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", errors="replace"))
    except error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {url}: {body[:800]}") from e


def _pick_model_id(models_json: dict[str, Any]) -> str:
    entries: list[dict[str, Any]] = []
    if isinstance(models_json.get("data"), list):
        entries = [x for x in models_json["data"] if isinstance(x, dict)]
    elif isinstance(models_json.get("models"), list):
        entries = [x for x in models_json["models"] if isinstance(x, dict)]

    ids: list[str] = []
    for e in entries:
        for k in ("id", "name", "model"):
            if e.get(k) is not None:
                ids.append(str(e[k]))
                break
    if not ids:
        raise RuntimeError("No model id found in /v1/models response")

    for mid in ids:
        low = mid.lower()
        if "brain" in low and "draft" not in low:
            return mid
    for mid in ids:
        if "draft" not in mid.lower():
            return mid
    return ids[0]


def _wait_server_ready(base_url: str, timeout_s: int = 120) -> str:
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout_s:
        try:
            j = _http_json("GET", base_url + "/v1/models", timeout=10)
            return _pick_model_id(j)
        except Exception as e:  # noqa: BLE001
            last = str(e)
            time.sleep(1.0)
    raise RuntimeError(f"Server not ready within {timeout_s}s. Last error: {last}")


def _run_chat_once(
    base_url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    repeat_penalty: float,
) -> tuple[float, float, float, float]:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "repeat_penalty": repeat_penalty,
    }
    t0 = time.perf_counter()
    resp = _http_json("POST", base_url + "/v1/chat/completions", payload, timeout=240)
    wall = time.perf_counter() - t0

    out_tok = float(resp.get("usage", {}).get("completion_tokens", 0.0))
    timings = resp.get("timings", {})
    pred_tps = float(timings.get("predicted_per_second", 0.0))
    wall_tps = (out_tok / wall) if wall > 0 else 0.0
    return wall, out_tok, pred_tps, wall_tps


def _run_chat_once_retry(
    base_url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    repeat_penalty: float,
    retries: int = 3,
) -> tuple[float, float, float, float]:
    last: Exception | None = None
    for i in range(max(1, retries)):
        try:
            return _run_chat_once(
                base_url=base_url,
                model_id=model_id,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                repeat_penalty=repeat_penalty,
            )
        except Exception as e:  # noqa: BLE001
            last = e
            # give server a moment to recover from transient connection resets
            time.sleep(0.8 * (i + 1))
    assert last is not None
    raise last


def _mean_sd(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(statistics.mean(xs)), float(statistics.stdev(xs))


def _launch_server(args: list[str]) -> subprocess.Popen[str]:
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        creationflags=creationflags,
    )


def _stop_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            time.sleep(1.0)
    except Exception:
        pass
    try:
        proc.terminate()
        proc.wait(timeout=8)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _benchmark_mode(
    *,
    llama_server: str,
    brain_path: str,
    draft_path: str | None,
    host: str,
    port: int,
    ctx: int,
    ngl: int,
    ngld: int,
    draft_max: int,
    draft_min: int,
    prompt: str,
    warmup: int,
    runs: int,
    max_tokens: int,
    temperature: float,
    repeat_penalty: float,
) -> dict[str, float]:
    cmd = [
        llama_server,
        "--model",
        brain_path,
        "--webui",
        "--jinja",
        "-c",
        str(ctx),
        "-ngl",
        str(ngl),
        "--host",
        host,
        "--port",
        str(port),
    ]
    if ngld >= 0:
        cmd += ["-ngld", str(ngld)]
    if draft_path:
        cmd += [
            "--model-draft",
            draft_path,
            "--draft-max",
            str(draft_max),
            "--draft-min",
            str(draft_min),
        ]

    base = f"http://{host}:{port}"
    proc = _launch_server(cmd)
    try:
        model_id = _wait_server_ready(base, timeout_s=180)
        for _ in range(max(0, warmup)):
            _run_chat_once_retry(base, model_id, prompt, max_tokens, temperature, repeat_penalty)

        walls: list[float] = []
        preds: list[float] = []
        wall_tps: list[float] = []
        out_toks: list[float] = []
        for _ in range(max(1, runs)):
            wall, out_tok, pred_tps, wtps = _run_chat_once_retry(
                base, model_id, prompt, max_tokens, temperature, repeat_penalty
            )
            walls.append(wall)
            out_toks.append(out_tok)
            preds.append(pred_tps)
            wall_tps.append(wtps)

        wall_m, wall_sd = _mean_sd(walls)
        pred_m, pred_sd = _mean_sd(preds)
        wtps_m, wtps_sd = _mean_sd(wall_tps)
        out_m, _ = _mean_sd(out_toks)
        return {
            "wall_mean_s": wall_m,
            "wall_sd_s": wall_sd,
            "predicted_tps_mean": pred_m,
            "predicted_tps_sd": pred_sd,
            "wall_tps_mean": wtps_m,
            "wall_tps_sd": wtps_sd,
            "out_tok_mean": out_m,
        }
    finally:
        _stop_server(proc)


def main() -> int:
    p = argparse.ArgumentParser(description="A/B benchmark: brain-only vs draft+brain on llama-server.")
    p.add_argument("--llama-server", type=str, default=r"C:\Users\SHIVANSH\Desktop\llama.cpp\build\bin\Release\llama-server.exe")
    p.add_argument("--brain", type=str, required=True, help="Path to Brain GGUF")
    p.add_argument("--draft", type=str, required=True, help="Path to Draft GGUF")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--ngld", type=int, default=0)
    p.add_argument("--draft-max", type=int, default=8)
    p.add_argument("--draft-min", type=int, default=2)
    p.add_argument("--prompt", type=str, default="Explain recursion in three short bullet points.")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--runs", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--repeat-penalty", type=float, default=1.15)
    p.add_argument("--target-speedup", type=float, default=2.0, help="Mark pass/fail threshold (e.g. 2.0)")
    args = p.parse_args()

    llama = os.path.abspath(args.llama_server)
    brain = os.path.abspath(args.brain)
    draft = os.path.abspath(args.draft)
    if not os.path.isfile(llama):
        print(f"llama-server not found: {llama}", file=sys.stderr)
        return 2
    if not os.path.isfile(brain):
        print(f"Brain GGUF not found: {brain}", file=sys.stderr)
        return 2
    if not os.path.isfile(draft):
        print(f"Draft GGUF not found: {draft}", file=sys.stderr)
        return 2

    print("=== A/B benchmark (same brain, draft off vs on) ===")
    print(f"Brain: {brain}")
    print(f"Draft: {draft}")
    print(f"Runs={args.runs}, Warmup={args.warmup}, MaxTokens={args.max_tokens}, Temp={args.temperature}, Repeat={args.repeat_penalty}")

    print("\n[1/2] Baseline: brain-only ...")
    base = _benchmark_mode(
        llama_server=llama,
        brain_path=brain,
        draft_path=None,
        host=args.host,
        port=args.port,
        ctx=args.ctx,
        ngl=args.ngl,
        ngld=args.ngld,
        draft_max=args.draft_max,
        draft_min=args.draft_min,
        prompt=args.prompt,
        warmup=args.warmup,
        runs=args.runs,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        repeat_penalty=args.repeat_penalty,
    )

    print("[2/2] SpinalCord-integrated: brain+draft ...")
    sc = _benchmark_mode(
        llama_server=llama,
        brain_path=brain,
        draft_path=draft,
        host=args.host,
        port=args.port,
        ctx=args.ctx,
        ngl=args.ngl,
        ngld=args.ngld,
        draft_max=args.draft_max,
        draft_min=args.draft_min,
        prompt=args.prompt,
        warmup=args.warmup,
        runs=args.runs,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        repeat_penalty=args.repeat_penalty,
    )

    wall_speedup = (base["wall_mean_s"] / sc["wall_mean_s"]) if sc["wall_mean_s"] > 0 else 0.0
    wall_tps_speedup = (sc["wall_tps_mean"] / base["wall_tps_mean"]) if base["wall_tps_mean"] > 0 else 0.0
    pred_tps_speedup = (sc["predicted_tps_mean"] / base["predicted_tps_mean"]) if base["predicted_tps_mean"] > 0 else 0.0

    print("\n=== RESULT ===")
    print(f"Baseline (brain-only): wall={base['wall_mean_s']:.3f}s ± {base['wall_sd_s']:.3f}, wall_tps={base['wall_tps_mean']:.1f}, predicted_tps={base['predicted_tps_mean']:.1f}")
    print(f"SpinalCord (brain+draft): wall={sc['wall_mean_s']:.3f}s ± {sc['wall_sd_s']:.3f}, wall_tps={sc['wall_tps_mean']:.1f}, predicted_tps={sc['predicted_tps_mean']:.1f}")
    print(f"Speedup (wall time): {wall_speedup:.2f}x")
    print(f"Speedup (wall tok/s): {wall_tps_speedup:.2f}x")
    print(f"Speedup (predicted tok/s): {pred_tps_speedup:.2f}x")

    verdict = "PASS" if wall_speedup >= args.target_speedup else "BELOW_TARGET"
    print(f"Target >= {args.target_speedup:.2f}x: {verdict}")
    print("Note: 10x is theoretical upper bound at very high acceptance; real-world depends on model alignment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
