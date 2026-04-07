#!/usr/bin/env python3
"""
Compare Brain-only vs speculative vs reflex under greedy vs sampled decoding.

Run from repo root (so `models/` and `train/` resolve):

  python scripts/diagnose_inference_quality.py
  python scripts/diagnose_inference_quality.py --prompt "What is 2+2?" --max-new-tokens 128
  python scripts/diagnose_inference_quality.py --modes brain,spec --greedy-only

See docs/INFERENCE_QUALITY_DEBUG.md for llama-server + dashboard parity checks.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

# Import from train/
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "train"))

from config import SpinalCordConfig, BrainConfig  # type: ignore
from model import SpinalCordLLM  # type: ignore
from tokenizer_sc import load_tokenizer_and_export  # type: ignore


DEFAULT_PROMPTS = [
    ("Q&A", "What is photosynthesis? Answer in one short paragraph."),
    ("Coding", "Write a minimal C program that prints Hello, world! Output only the code."),
]


def load_model(device: str) -> tuple[SpinalCordLLM, object, BrainConfig]:
    cfg = SpinalCordConfig()
    bundle, _ = load_tokenizer_and_export(expected_vocab_size=cfg.brain.vocab_size)
    brain_ckpt = os.path.join(_ROOT, "models", "scbrain_best.pt")
    draft_ckpt = os.path.join(_ROOT, "models", "scdraft_best.pt")

    if not os.path.isfile(brain_ckpt):
        raise FileNotFoundError(f"Missing {brain_ckpt}")
    if not os.path.isfile(draft_ckpt):
        raise FileNotFoundError(f"Missing {draft_ckpt}")

    brain_state = torch.load(brain_ckpt, map_location=device, weights_only=False)
    draft_state = torch.load(draft_ckpt, map_location=device, weights_only=False)

    brain_cfg: BrainConfig = brain_state["cfg"]
    draft_cfg = draft_state["cfg"]

    model = SpinalCordLLM(draft_cfg, brain_cfg).to(device)
    model.brain.load_state_dict(brain_state["model_state"])
    model.draft.load_state_dict(draft_state["model_state"])
    model.eval()
    return model, bundle, brain_cfg


def decode_new(bundle, input_ids: torch.Tensor, full_ids: torch.Tensor) -> str:
    inp = input_ids[0].tolist()
    out = full_ids[0].tolist()
    return bundle.decode(out[len(inp) :])


def run_matrix(
    model: SpinalCordLLM,
    bundle,
    label: str,
    prompt: str,
    *,
    device: str,
    max_new_tokens: int,
    modes: set[str],
    do_greedy: bool,
    do_sampled: bool,
    sampled_temp: float,
    top_k: int,
    top_p: float,
    reflex_verbose: bool,
) -> None:
    input_ids = torch.tensor([bundle.encode(prompt)], dtype=torch.long, device=device)
    print("\n" + "=" * 72)
    print(f"[{label}] prompt ({len(prompt)} chars)")
    print("-" * 72)
    print(prompt[:2000] + ("…" if len(prompt) > 2000 else ""))
    print("=" * 72)

    def section(title: str) -> None:
        print(f"\n>>> {title}")
        print("-" * 48)

    if "brain" in modes and do_greedy:
        section("Brain-only | greedy (argmax, temp<=0)")
        t0 = time.perf_counter()
        out = model.generate_brain_only(
            input_ids,
            max_new_tokens=max_new_tokens,
            brain_device=device,
            verbose=False,
            temperature=0.0,
        )
        dt = time.perf_counter() - t0
        text = decode_new(bundle, input_ids, out)
        print(text[:4000])
        if len(text) > 4000:
            print("… [truncated]")
        print(f"(wall {dt:.2f}s)")

    if "brain" in modes and do_sampled:
        section(f"Brain-only | sampled (temp={sampled_temp}, top_k={top_k}, top_p={top_p})")
        t0 = time.perf_counter()
        out = model.generate_brain_only(
            input_ids,
            max_new_tokens=max_new_tokens,
            brain_device=device,
            verbose=False,
            temperature=sampled_temp,
            top_k=top_k,
            top_p=top_p,
        )
        dt = time.perf_counter() - t0
        text = decode_new(bundle, input_ids, out)
        print(text[:4000])
        if len(text) > 4000:
            print("… [truncated]")
        print(f"(wall {dt:.2f}s)")

    if "spec" in modes and do_greedy:
        section("Speculative generate() | greedy (temp=0 → near-greedy in draft/brain)")
        t0 = time.perf_counter()
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            brain_device=device,
            draft_device=device,
            verbose=reflex_verbose,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
        )
        dt = time.perf_counter() - t0
        text = decode_new(bundle, input_ids, out)
        print(text[:4000])
        if len(text) > 4000:
            print("… [truncated]")
        print(f"(wall {dt:.2f}s)")

    if "spec" in modes and do_sampled:
        section(f"Speculative generate() | sampled (temp={sampled_temp})")
        t0 = time.perf_counter()
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            brain_device=device,
            draft_device=device,
            verbose=reflex_verbose,
            temperature=sampled_temp,
            top_k=top_k,
            top_p=top_p,
        )
        dt = time.perf_counter() - t0
        text = decode_new(bundle, input_ids, out)
        print(text[:4000])
        if len(text) > 4000:
            print("… [truncated]")
        print(f"(wall {dt:.2f}s)")

    if "reflex" in modes and do_greedy:
        section("generate_reflex() | greedy")
        t0 = time.perf_counter()
        out = model.generate_reflex(
            input_ids,
            max_new_tokens=max_new_tokens,
            brain_device=device,
            draft_device=device,
            accept_rate_threshold=0.65,
            consecutive_bad_rounds_to_fallback=2,
            recover_speculative_after_brain_tokens=24,
            rolling_accept_window=4,
            verbose=reflex_verbose,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
        )
        dt = time.perf_counter() - t0
        text = decode_new(bundle, input_ids, out)
        print(text[:4000])
        if len(text) > 4000:
            print("… [truncated]")
        print(f"(wall {dt:.2f}s)")

    if "reflex" in modes and do_sampled:
        section(f"generate_reflex() | sampled (temp={sampled_temp})")
        t0 = time.perf_counter()
        out = model.generate_reflex(
            input_ids,
            max_new_tokens=max_new_tokens,
            brain_device=device,
            draft_device=device,
            accept_rate_threshold=0.65,
            consecutive_bad_rounds_to_fallback=2,
            recover_speculative_after_brain_tokens=24,
            rolling_accept_window=4,
            verbose=reflex_verbose,
            temperature=sampled_temp,
            top_k=top_k,
            top_p=top_p,
        )
        dt = time.perf_counter() - t0
        text = decode_new(bundle, input_ids, out)
        print(text[:4000])
        if len(text) > 4000:
            print("… [truncated]")
        print(f"(wall {dt:.2f}s)")


def main() -> None:
    p = argparse.ArgumentParser(description="SpinalCord inference quality matrix (PyTorch).")
    p.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Single prompt (if empty, uses built-in default prompts).",
    )
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument(
        "--modes",
        type=str,
        default="brain,spec,reflex",
        help="Comma-separated: brain, spec, reflex (default: all).",
    )
    p.add_argument("--greedy-only", action="store_true", help="Only temp=0 / greedy-style runs.")
    p.add_argument("--sampled-only", action="store_true", help="Only sampled runs.")
    p.add_argument("--sampled-temp", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument(
        "--verbose-steps",
        action="store_true",
        help="Print per-step speculative stats (noisy).",
    )
    args = p.parse_args()

    if args.greedy_only and args.sampled_only:
        print("Choose at most one of --greedy-only and --sampled-only.", file=sys.stderr)
        sys.exit(2)

    modes = {m.strip().lower() for m in args.modes.split(",") if m.strip()}
    valid = {"brain", "spec", "reflex"}
    bad = modes - valid
    if bad:
        print(f"Unknown modes: {bad}. Use: {valid}", file=sys.stderr)
        sys.exit(2)

    do_greedy = not args.sampled_only
    do_sampled = not args.greedy_only

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model, bundle, _ = load_model(device)

    prompts: list[tuple[str, str]]
    if args.prompt.strip():
        prompts = [("Custom", args.prompt.strip())]
    else:
        prompts = list(DEFAULT_PROMPTS)

    for label, pr in prompts:
        run_matrix(
            model,
            bundle,
            label,
            pr,
            device=device,
            max_new_tokens=args.max_new_tokens,
            modes=modes,
            do_greedy=do_greedy,
            do_sampled=do_sampled,
            sampled_temp=args.sampled_temp,
            top_k=args.top_k,
            top_p=args.top_p,
            reflex_verbose=args.verbose_steps,
        )

    print("\nDone. Interpret: if **brain-only** is bad but checkpoints look fine, suspect tokenizer/vocab;")
    print("if brain-only is OK but **spec/reflex** fails, tune draft/distill or disable draft in server.")


if __name__ == "__main__":
    main()
