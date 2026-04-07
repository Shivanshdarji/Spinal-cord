"""
SpinalCord — Speed benchmark (local) + optional Anthropic Opus comparison

Local (always):
  - Brain-only: autoregressive next-token from SpinalCordBrain
  - SpinalCord: SpinalCordLLM.generate() (Draft + Brain speculative)

Optional API (if ANTHROPIC_API_KEY is set):
  - Same prompt and max_tokens. Sampling: many models (e.g. Opus 4.x) allow **either** temperature **or**
    top_p, not both — we send one (default: temperature only; use --api-use-top-p-only for top_p only).
  - Same warmup + run count as local by default (--api-warmup / --api-runs override).
  - Uses streaming to report: wall tok/s, time-to-first-token (TTFT), and decode-phase tok/s
    (tokens emitted after the first chunk — closer to "generation speed" vs one-shot wall).

Fairness limits (unavoidable):
  - Different hardware (your GPU vs Anthropic servers) — compare methodology, not absolute $.
  - API includes network RTT; TTFT + decode metrics separate fixed latency from output rate.
  - Local tokenizer vs Claude — prompt/output token counts differ; we report both where possible.

Recommended “fairest” run:
  python -u benchmark_speed.py --greedy --fair-mirror-api --runs 10 --warmup 3 --max-new-tokens 128

Usage:
  python -u benchmark_speed.py
  python -u benchmark_speed.py --max-new-tokens 64 --runs 5
  python -u benchmark_speed.py --greedy --fair-mirror-api
  $env:ANTHROPIC_API_KEY="..."; $env:ANTHROPIC_MODEL="claude-opus-4-6"; python -u benchmark_speed.py
"""

from __future__ import annotations

import argparse
import os
import random
import warnings
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "train"))

from config import BrainConfig  # type: ignore
from model import SpinalCordBrain, SpinalCordLLM  # type: ignore
from tokenizer_sc import load_tokenizer_and_export  # type: ignore


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _set_seeds(seed: int) -> None:
    """Reproducible sampling across runs (best-effort; some GPU ops remain nondeterministic)."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def brain_only_generate(
    brain: SpinalCordBrain,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    device: str,
    temperature: float,
    top_k: int,
    top_p: float,
    timing_out: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """Match test_pytorch.py style: argmax or sample from brain last logits."""
    from model import _sample_from_logits  # type: ignore

    generated = input_ids.clone()
    if timing_out is not None:
        timing_out.clear()
        _sync_cuda()
        t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = brain(generated.to(device))
            next_token, _ = _sample_from_logits(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            generated = torch.cat([generated, next_token.to(generated.device)], dim=-1)
            if timing_out is not None and "ttft" not in timing_out:
                _sync_cuda()
                timing_out["ttft"] = time.perf_counter() - t0
    if timing_out is not None:
        _sync_cuda()
        timing_out["total"] = time.perf_counter() - t0
        timing_out["tokens_out"] = float(max_new_tokens)
    return generated


def _mean_stdev(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return m, sd


def run_local_benchmark(
    *,
    device: str,
    prompt: str,
    max_new_tokens: int,
    runs: int,
    warmup: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
) -> Tuple[dict, dict, dict]:
    """
    Returns (brain_stats, spinal_stats, meta) with TTFT + decode tok/s aligned to API methodology.
    """
    brain_ckpt = os.path.join("models", "scbrain_best.pt")
    draft_ckpt = os.path.join("models", "scdraft_best.pt")

    brain_state = torch.load(brain_ckpt, map_location=device, weights_only=False)
    draft_state = torch.load(draft_ckpt, map_location=device, weights_only=False)
    brain_cfg: BrainConfig = brain_state["cfg"]
    draft_cfg = draft_state["cfg"]

    bundle, _ = load_tokenizer_and_export(expected_vocab_size=brain_cfg.vocab_size)
    encode, _ = bundle.encode, bundle.decode
    prompt_ids = encode(prompt)
    prompt_tokens = len(prompt_ids)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    brain = SpinalCordBrain(brain_cfg).to(device)
    brain.load_state_dict(brain_state["model_state"])
    brain.eval()

    llm = SpinalCordLLM(draft_cfg, brain_cfg).to(device)
    llm.brain.load_state_dict(brain_state["model_state"])
    llm.draft.load_state_dict(draft_state["model_state"])
    llm.eval()

    p0 = next(brain.parameters())
    meta = {
        "device": device,
        "prompt_tokens": prompt_tokens,
        "dtype": str(p0.dtype),
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" and torch.cuda.is_available() else None,
    }

    def bench_brain() -> None:
        _ = brain_only_generate(
            brain, input_ids, max_new_tokens, device, temperature, top_k, top_p
        )

    def bench_spinal() -> None:
        _ = llm.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            brain_device=device,
            draft_device=device,
            verbose=False,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    for w in range(warmup):
        _set_seeds(seed + 10_000 + w)
        bench_brain()
        _set_seeds(seed + 10_000 + w)
        bench_spinal()

    brain_wall: List[float] = []
    brain_ttft: List[float] = []
    brain_decode_tps: List[float] = []
    brain_wall_tps: List[float] = []

    spinal_wall: List[float] = []
    spinal_ttft: List[float] = []
    spinal_decode_tps: List[float] = []
    spinal_wall_tps: List[float] = []

    for i in range(runs):
        _set_seeds(seed + i)
        tb: Dict[str, float] = {}
        brain_only_generate(
            brain, input_ids, max_new_tokens, device, temperature, top_k, top_p, timing_out=tb
        )
        wall_b = float(tb["total"])
        ttft_b = float(tb["ttft"])
        dec_b = max(wall_b - ttft_b, 1e-9)
        tok_b = float(tb["tokens_out"])
        brain_wall.append(wall_b)
        brain_ttft.append(ttft_b)
        brain_decode_tps.append(tok_b / dec_b)
        brain_wall_tps.append(tok_b / wall_b)

        _set_seeds(seed + i)
        ts: Dict[str, Any] = {}
        llm.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            brain_device=device,
            draft_device=device,
            verbose=False,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            _timing_out=ts,
        )
        wall_s = float(ts["total"])
        ttft_s = float(ts.get("ttft", wall_s))
        dec_s = max(wall_s - ttft_s, 1e-9)
        tok_s = float(ts["tokens_out"])
        spinal_wall.append(wall_s)
        spinal_ttft.append(ttft_s)
        spinal_decode_tps.append(tok_s / dec_s)
        spinal_wall_tps.append(tok_s / wall_s)

    def pack(name: str, wall: List[float], ttft: List[float], dec_tps: List[float], w_tps: List[float]) -> dict:
        wm, ws = _mean_stdev(wall)
        tm, ts = _mean_stdev(ttft)
        dm, dsd = _mean_stdev(dec_tps)
        wpm, wps = _mean_stdev(w_tps)
        return {
            "name": name,
            "seconds_mean": wm,
            "seconds_stdev": ws,
            "tok_per_s_mean": wpm,
            "tok_per_s_stdev": wps,
            "ttft_seconds_mean": tm,
            "ttft_seconds_stdev": ts,
            "decode_tok_per_s_mean": dm,
            "decode_tok_per_s_stdev": dsd,
        }

    b = pack("Brain-only (local)", brain_wall, brain_ttft, brain_decode_tps, brain_wall_tps)
    s = pack("SpinalCord generate (local)", spinal_wall, spinal_ttft, spinal_decode_tps, spinal_wall_tps)
    b["speedup_vs_brain"] = float(b["seconds_mean"] / s["seconds_mean"]) if s["seconds_mean"] > 0 else 0.0
    return b, s, meta


def _anthropic_sampling_kwargs(*, temperature: float, top_p: float, api_use_top_p_only: bool) -> dict:
    """
    Anthropic Messages API: some models reject `temperature` and `top_p` together (400).
    Send exactly one of them.
    """
    if api_use_top_p_only:
        return {"top_p": top_p}
    return {"temperature": temperature}


def _anthropic_one_stream(
    client: object,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    api_use_top_p_only: bool,
) -> Tuple[float, Optional[float], Optional[float], int, int]:
    """
    One streamed completion. Returns:
      wall_s, ttft_s (None if no chunk), decode_s (wall minus TTFT), output_tokens, input_tokens
    """
    t_wall0 = time.perf_counter()
    t_first: Optional[float] = None
    msg = None
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*[Dd]eprecat.*",
            category=DeprecationWarning,
        )
        # `client.messages.stream` exists in anthropic>=0.34
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **_anthropic_sampling_kwargs(
                temperature=temperature, top_p=top_p, api_use_top_p_only=api_use_top_p_only
            ),
        ) as stream:
            for _ in stream.text_stream:
                if t_first is None:
                    t_first = time.perf_counter()
            msg = stream.get_final_message()
    t_wall1 = time.perf_counter()
    wall_s = t_wall1 - t_wall0

    out_tokens = 0
    in_tokens = 0
    if msg is not None:
        usage = getattr(msg, "usage", None)
        out_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage is not None else 0
        in_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage is not None else 0
    if out_tokens <= 0:
        out_text = ""
        if msg is not None:
            for block in msg.content:
                if getattr(block, "type", None) == "text":
                    out_text += block.text
        out_tokens = max(1, len(out_text) // 4)

    ttft_s = (t_first - t_wall0) if t_first is not None else None
    decode_s: Optional[float] = None
    if t_first is not None:
        decode_s = max(t_wall1 - t_first, 1e-9)
    return wall_s, ttft_s, decode_s, out_tokens, in_tokens


def run_anthropic_benchmark(
    *,
    prompt: str,
    max_tokens: int,
    model: str,
    temperature: float,
    top_p: float,
    runs: int,
    warmup: int,
    use_stream: bool,
    api_use_top_p_only: bool,
) -> Optional[dict]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic  # type: ignore
    except ImportError:
        print(
            "[API] anthropic package not installed. Install: pip install anthropic",
            file=sys.stderr,
        )
        return None

    client = anthropic.Anthropic(api_key=key)
    if use_stream and not hasattr(client.messages, "stream"):
        print(
            "[API] messages.stream unavailable; using messages.create. "
            "pip install 'anthropic>=0.34' for TTFT / decode metrics.",
            file=sys.stderr,
        )
        use_stream = False

    def run_create_only() -> Tuple[float, Optional[float], Optional[float], int, int]:
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*[Dd]eprecat.*",
                category=DeprecationWarning,
            )
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                **_anthropic_sampling_kwargs(
                    temperature=temperature, top_p=top_p, api_use_top_p_only=api_use_top_p_only
                ),
            )
        elapsed = time.perf_counter() - t0
        usage = getattr(msg, "usage", None)
        out_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage is not None else 0
        in_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage is not None else 0
        if out_tokens <= 0:
            out_text = ""
            for block in msg.content:
                if getattr(block, "type", None) == "text":
                    out_text += block.text
            out_tokens = max(1, len(out_text) // 4)
        return elapsed, None, None, out_tokens, in_tokens

    try:
        for _ in range(max(0, warmup)):
            if use_stream:
                _anthropic_one_stream(
                    client,
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    api_use_top_p_only=api_use_top_p_only,
                )
            else:
                run_create_only()

        wall_times: List[float] = []
        ttft_times: List[float] = []
        decode_tok_rates: List[float] = []
        out_tok_list: List[int] = []
        in_tok_list: List[int] = []
        wall_tok_per_s: List[float] = []

        for _ in range(max(1, runs)):
            if use_stream:
                wall_s, ttft_s, decode_s, out_tokens, in_tokens = _anthropic_one_stream(
                    client,
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    api_use_top_p_only=api_use_top_p_only,
                )
                wall_times.append(wall_s)
                if wall_s > 0:
                    wall_tok_per_s.append(float(out_tokens) / wall_s)
                if ttft_s is not None:
                    ttft_times.append(ttft_s)
                if decode_s is not None and decode_s > 0 and out_tokens > 0:
                    decode_tok_rates.append(float(out_tokens) / decode_s)
                out_tok_list.append(out_tokens)
                in_tok_list.append(in_tokens)
            else:
                wall_s, _, _, out_tokens, in_tokens = run_create_only()
                wall_times.append(wall_s)
                if wall_s > 0:
                    wall_tok_per_s.append(float(out_tokens) / wall_s)
                out_tok_list.append(out_tokens)
                in_tok_list.append(in_tokens)

    except Exception as e:
        err = str(e).lower()
        is_404 = "404" in str(e) or "not_found" in err
        is_billing = (
            "credit balance" in err
            or "too low" in err
            or "billing" in err
            or "purchase credits" in err
        )
        is_temp_top_p = "temperature" in err and "top_p" in err and "cannot both" in err
        print(f"\n[API] Anthropic request failed: {e}", file=sys.stderr)
        if is_temp_top_p:
            print(
                "      This model requires temperature OR top_p, not both. "
                "This script uses one parameter by default (temperature). "
                "If you still see this, try: --api-use-top-p-only\n",
                file=sys.stderr,
            )
        if is_billing:
            print(
                "      Your Anthropic account has no/low credits — not a bug in this script.\n"
                "      Add credits: https://console.anthropic.com/ → Plans & Billing\n"
                "      Until then: python -u benchmark_speed.py --skip-api (local only).\n",
                file=sys.stderr,
            )
        elif is_404:
            print(
                f"      Model id {model!r} is invalid or not enabled for this API key / region.\n"
                f"      Safe default: $env:ANTHROPIC_MODEL=\"claude-3-5-sonnet-20241022\"\n"
                f"      For Opus / newer tiers, copy the exact id from Anthropic Console → Workbench\n"
                f"      or https://docs.anthropic.com/en/docs/about-claude/models (not 'latest' aliases).\n",
                file=sys.stderr,
            )
        return None

    def mean(xs: List[float]) -> float:
        return statistics.mean(xs) if xs else 0.0

    def stdev(xs: List[float]) -> float:
        return statistics.stdev(xs) if len(xs) > 1 else 0.0

    wall_mean = mean(wall_times)
    out_mean = mean([float(t) for t in out_tok_list]) if out_tok_list else 0.0
    in_mean = mean([float(t) for t in in_tok_list]) if in_tok_list and any(in_tok_list) else None
    short_runs = sum(1 for t in out_tok_list if t < max_tokens)

    sampling_note = (
        f"top_p={top_p} only" if api_use_top_p_only else f"temperature={temperature} only"
    )
    result: dict = {
        "name": f"Anthropic API ({model})",
        "runs": runs,
        "warmup": warmup,
        "api_sampling": sampling_note,
        "wall_seconds_mean": wall_mean,
        "wall_seconds_stdev": stdev(wall_times),
        "tok_per_s_wall_mean": mean(wall_tok_per_s) if wall_tok_per_s else 0.0,
        "tok_per_s_wall_stdev": stdev(wall_tok_per_s) if len(wall_tok_per_s) > 1 else 0.0,
        "output_tokens_mean": out_mean,
        "input_tokens_mean": in_mean,
        "output_short_runs": short_runs,
        "use_stream": use_stream,
        "ttft_seconds_mean": mean(ttft_times) if ttft_times else None,
        "ttft_seconds_stdev": stdev(ttft_times) if len(ttft_times) > 1 else 0.0,
        "decode_tok_per_s_mean": mean(decode_tok_rates) if decode_tok_rates else None,
        "decode_tok_per_s_stdev": stdev(decode_tok_rates) if len(decode_tok_rates) > 1 else 0.0,
    }
    return result


def _print_fair_comparison_table(
    *,
    b: dict,
    s: dict,
    api: Optional[dict],
    meta: dict,
) -> None:
    """Side-by-side metrics that use the same definitions as the Anthropic stream benchmark."""
    print()
    print("=== Fair comparison (same definitions) ===")
    print(
        "  Decode tok/s = output_tokens / (wall_time - time_to_first_token). "
        "Matches streamed API 'decode phase' vs local time after first generated token."
    )
    print(
        f"  Local prompt length: {meta['prompt_tokens']} tokens (SpinalCord tokenizer). "
        + (
            f"API prompt (usage): ~{api['input_tokens_mean']:.0f} tokens."
            if api and api.get("input_tokens_mean") is not None
            else "API prompt: (enable streaming + usage.input_tokens)."
        )
    )
    print(
        f"  Hardware: device={meta['device']} dtype={meta['dtype']}"
        + (f" | {meta['gpu_name']}" if meta.get("gpu_name") else "")
    )
    print()
    rows = [
        ("Metric", "Brain (local)", "SpinalCord (local)", "Anthropic (API)" if api else "-"),
        ("Wall tok/s", f"{b['tok_per_s_mean']:.1f} +/- {b['tok_per_s_stdev']:.1f}", f"{s['tok_per_s_mean']:.1f} +/- {s['tok_per_s_stdev']:.1f}", ""),
        ("TTFT (s)", f"{b['ttft_seconds_mean']:.4f} +/- {b['ttft_seconds_stdev']:.4f}", f"{s['ttft_seconds_mean']:.4f} +/- {s['ttft_seconds_stdev']:.4f}", ""),
        ("Decode tok/s (*)", f"{b['decode_tok_per_s_mean']:.1f} +/- {b['decode_tok_per_s_stdev']:.1f}", f"{s['decode_tok_per_s_mean']:.1f} +/- {s['decode_tok_per_s_stdev']:.1f}", ""),
    ]
    if api:
        rows[1] = (
            rows[1][0],
            rows[1][1],
            rows[1][2],
            f"{api['tok_per_s_wall_mean']:.1f} +/- {api['tok_per_s_wall_stdev']:.1f}",
        )
        rows[2] = (
            rows[2][0],
            rows[2][1],
            rows[2][2],
            f"{api['ttft_seconds_mean']:.4f} +/- {api['ttft_seconds_stdev']:.4f}"
            if api.get("ttft_seconds_mean") is not None
            else "n/a",
        )
        dmean = api.get("decode_tok_per_s_mean")
        dsd = api.get("decode_tok_per_s_stdev")
        rows[3] = (
            rows[3][0],
            rows[3][1],
            rows[3][2],
            f"{dmean:.1f} +/- {dsd:.1f}" if dmean is not None else "n/a",
        )
    else:
        for i in range(1, 4):
            rows[i] = (*rows[i][:3], "-")

    colw = [22, 22, 26, 22]
    for ri, row in enumerate(rows):
        cells = [str(x)[: colw[i] - 1] for i, x in enumerate(row)]
        print("  " + " | ".join(c.ljust(colw[i]) for i, c in enumerate(cells)))
        if ri == 0:
            print("  " + "-+-".join("-" * colw[i] for i in range(4)))
    print("  (*) Primary row for 'generation rate' vs Opus decode-phase tok/s.")
    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default="Explain what a neural network is in two sentences.")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument(
        "--greedy",
        action="store_true",
        help="Set temperature=0 for local + Anthropic (best-effort apples-to-apples speed; API still stochastic edge cases).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed(s): run i uses seed+i for reproducible sampling (local only).",
    )
    p.add_argument(
        "--fair-mirror-api",
        action="store_true",
        help="Set local top_k=0 to mirror Anthropic (no top-k truncation). Recommended for fairest sampling match.",
    )
    p.add_argument(
        "--no-fair-table",
        action="store_true",
        help="Skip the side-by-side fair comparison table at the end.",
    )
    p.add_argument(
        "--anthropic-model",
        type=str,
        default=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        help=(
            "Anthropic Messages API model id (exact string from Console / docs). "
            "Default: claude-3-5-sonnet-20241022. Older snapshots (e.g. claude-3-opus-20240229) "
            "may 404 or be deprecated — use a current id from your account."
        ),
    )
    p.add_argument(
        "--skip-api",
        action="store_true",
        help="Only run local Brain vs SpinalCord (no Anthropic call).",
    )
    p.add_argument(
        "--api-runs",
        type=int,
        default=None,
        help="Number of timed Anthropic runs (default: same as --runs).",
    )
    p.add_argument(
        "--api-warmup",
        type=int,
        default=None,
        help="Anthropic warmup runs before timing (default: same as --warmup).",
    )
    p.add_argument(
        "--api-no-stream",
        action="store_true",
        help="Use blocking messages.create instead of streaming (no TTFT / decode-phase tok/s).",
    )
    p.add_argument(
        "--api-use-top-p-only",
        action="store_true",
        help=(
            "Send only top_p to Anthropic (omit temperature). Default sends only temperature. "
            "Required for some models that forbid both; Opus 4.x rejects both together."
        ),
    )
    args = p.parse_args()

    temperature = 0.0 if args.greedy else args.temperature
    top_k_eff = 0 if args.fair_mirror_api else args.top_k
    api_runs = args.api_runs if args.api_runs is not None else args.runs
    api_warmup = args.api_warmup if args.api_warmup is not None else args.warmup
    use_api_stream = not args.api_no_stream

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== SpinalCord local benchmark ===\nDevice: {device}\nPrompt: {args.prompt[:80]}...\n")
    print(
        f"max_new_tokens={args.max_new_tokens} runs={args.runs} warmup={args.warmup} seed={args.seed} "
        f"temperature={temperature} top_p={args.top_p} top_k={top_k_eff}"
        + (" (greedy)" if args.greedy else "")
        + (" | --fair-mirror-api: local top_k=0" if args.fair_mirror_api else "")
        + "\n"
    )

    b, s, meta = run_local_benchmark(
        device=device,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        runs=args.runs,
        warmup=args.warmup,
        temperature=temperature,
        top_k=top_k_eff,
        top_p=args.top_p,
        seed=args.seed,
    )

    print(f"{b['name']}")
    print(f"  wall time: {b['seconds_mean']:.3f}s +/- {b['seconds_stdev']:.3f}s")
    print(f"  wall tok/s: {b['tok_per_s_mean']:.1f} +/- {b['tok_per_s_stdev']:.1f}")
    print(f"  TTFT (first token): {b['ttft_seconds_mean']:.4f}s +/- {b['ttft_seconds_stdev']:.4f}s")
    print(f"  decode tok/s: {b['decode_tok_per_s_mean']:.1f} +/- {b['decode_tok_per_s_stdev']:.1f}")
    print()
    print(f"{s['name']}")
    print(f"  wall time: {s['seconds_mean']:.3f}s +/- {s['seconds_stdev']:.3f}s")
    print(f"  wall tok/s: {s['tok_per_s_mean']:.1f} +/- {s['tok_per_s_stdev']:.1f}")
    print(f"  TTFT (first output): {s['ttft_seconds_mean']:.4f}s +/- {s['ttft_seconds_stdev']:.4f}s")
    print(f"  decode tok/s: {s['decode_tok_per_s_mean']:.1f} +/- {s['decode_tok_per_s_stdev']:.1f}")
    print()
    print(f"  Local speedup (Brain wall / SpinalCord wall): {b['speedup_vs_brain']:.2f}x")
    print()

    api = None
    if not args.skip_api:
        api = run_anthropic_benchmark(
            prompt=args.prompt,
            max_tokens=args.max_new_tokens,
            model=args.anthropic_model,
            temperature=temperature,
            top_p=args.top_p,
            runs=api_runs,
            warmup=api_warmup,
            use_stream=use_api_stream,
            api_use_top_p_only=args.api_use_top_p_only,
        )
    if api:
        print("=== Anthropic comparison (methodology-matched) ===")
        loc_bits = []
        if args.greedy:
            loc_bits.append("temperature=0")
        else:
            loc_bits.append(f"temperature={temperature}")
        loc_bits.append(f"top_p={args.top_p}")
        loc_bits.append(f"top_k={top_k_eff}")
        print(
            f"  Same prompt, max_tokens={args.max_new_tokens}. "
            f"API sampling: {api.get('api_sampling', '?')}. "
            f"Local sampling: {', '.join(loc_bits)}. "
            "Anthropic API has no top_k."
        )
        print(f"  Timed runs={api['runs']} warmup={api['warmup']} stream={api['use_stream']}")
        print(f"{api['name']}")
        print(
            f"  Wall time: {api['wall_seconds_mean']:.3f}s +/- {api['wall_seconds_stdev']:.3f}s  "
            f"-> {api['tok_per_s_wall_mean']:.1f} tok/s +/- {api['tok_per_s_wall_stdev']:.1f} (output / wall)"
        )
        print(f"  Mean output_tokens (API usage): {api['output_tokens_mean']:.1f}")
        if api["output_short_runs"]:
            print(
                f"  Note: {api['output_short_runs']} run(s) stopped before max_tokens (EOS). "
                f"Local always generated {args.max_new_tokens} new tokens; API averaged "
                f"{api['output_tokens_mean']:.1f} — workloads differ; decode tok/s is only loosely comparable. "
                "For a fairer match, use --prompt that demands a long answer (e.g. several paragraphs) "
                "so Opus uses most of max_tokens."
            )
        if api["use_stream"] and api["ttft_seconds_mean"] is not None:
            print(
                f"  TTFT (first streamed chunk): {api['ttft_seconds_mean']:.3f}s +/- "
                f"{api['ttft_seconds_stdev']:.3f}s  (network + queue + prefill)"
            )
        if api["decode_tok_per_s_mean"] is not None:
            print(
                f"  Decode-phase tok/s: {api['decode_tok_per_s_mean']:.1f} +/- {api['decode_tok_per_s_stdev']:.1f}  "
                "(output_tokens / time after first chunk — closest to 'raw emit rate')"
            )
        print("  Interpretation: compare local tok/s to **decode-phase** for output rate; wall includes TTFT.")
        if api.get("input_tokens_mean") is not None:
            print(f"  API input_tokens (usage, mean): {api['input_tokens_mean']:.1f}")
    elif args.skip_api:
        print("=== Anthropic (skipped) ===")
        print("  Re-run without --skip-api to compare with Claude API.")
    else:
        print("=== Anthropic (optional) ===")
        if os.environ.get("ANTHROPIC_API_KEY"):
            print("  API key is set but the request failed or returned nothing — see messages above.")
        else:
            print("  Set ANTHROPIC_API_KEY and pip install anthropic to compare with Claude.")
        print(f"  Example model: $env:ANTHROPIC_MODEL=\"claude-opus-4-6\"")
    if not args.no_fair_table:
        _print_fair_comparison_table(b=b, s=s, api=api if not args.skip_api else None, meta=meta)
    print()


if __name__ == "__main__":
    main()
