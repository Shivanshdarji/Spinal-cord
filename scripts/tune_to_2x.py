#!/usr/bin/env python3
"""
Search for speculative settings that approach/exceed 2x on current hardware.

This wraps scripts/benchmark_serverpack_ab.py over a small grid and prints
the top configs by wall-time speedup.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class Trial:
    max_tokens: int
    ngld: int
    draft_max: int
    draft_min: int
    speedup: float
    out: str


def parse_speedup(text: str) -> float:
    m = re.search(r"Speedup \(wall time\):\s*([0-9.]+)x", text)
    if not m:
        return 0.0
    try:
        return float(m.group(1))
    except Exception:
        return 0.0


def run_trial(root: str, args: argparse.Namespace, max_tokens: int, ngld: int, draft_max: int, draft_min: int) -> Trial:
    cmd = [
        sys.executable,
        os.path.join(root, "scripts", "benchmark_serverpack_ab.py"),
        "--brain",
        args.brain,
        "--draft",
        args.draft,
        "--llama-server",
        args.llama_server,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--runs",
        str(args.runs),
        "--warmup",
        str(args.warmup),
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        str(args.temperature),
        "--repeat-penalty",
        str(args.repeat_penalty),
        "--target-speedup",
        str(args.target_speedup),
        "--ngl",
        str(args.ngl),
        "--ngld",
        str(ngld),
        "--draft-max",
        str(draft_max),
        "--draft-min",
        str(draft_min),
        "--ctx",
        str(args.ctx),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return Trial(
        max_tokens=max_tokens,
        ngld=ngld,
        draft_max=draft_max,
        draft_min=draft_min,
        speedup=parse_speedup(out),
        out=out,
    )


def main() -> int:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    p = argparse.ArgumentParser(description="Tune speculative config toward >=2x speedup.")
    p.add_argument("--llama-server", default=r"C:\Users\SHIVANSH\Desktop\llama.cpp\build\bin\Release\llama-server.exe")
    p.add_argument("--brain", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090, help="Use non-8080 to avoid conflicts with dashboard")
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--runs", type=int, default=4)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--repeat-penalty", type=float, default=1.15)
    p.add_argument("--target-speedup", type=float, default=2.0)
    args = p.parse_args()

    token_grid = [128, 256, 512]
    ngld_grid = [0, 99]
    draft_max_grid = [4, 6, 8]
    draft_min_grid = [1, 2]

    trials: list[Trial] = []
    total = len(token_grid) * len(ngld_grid) * len(draft_max_grid) * len(draft_min_grid)
    i = 0
    for t in token_grid:
        for ngld in ngld_grid:
            for dmax in draft_max_grid:
                for dmin in draft_min_grid:
                    if dmin > dmax:
                        continue
                    i += 1
                    print(f"[{i}/{total}] max_tokens={t} ngld={ngld} draft_max={dmax} draft_min={dmin}")
                    tr = run_trial(root, args, t, ngld, dmax, dmin)
                    print(f"  -> speedup={tr.speedup:.2f}x")
                    trials.append(tr)

    if not trials:
        print("No trials executed.")
        return 2

    trials.sort(key=lambda x: x.speedup, reverse=True)
    best = trials[0]

    print("\n=== TOP CONFIGS ===")
    for tr in trials[:8]:
        print(
            f"speedup={tr.speedup:.2f}x  "
            f"max_tokens={tr.max_tokens} ngld={tr.ngld} draft_max={tr.draft_max} draft_min={tr.draft_min}"
        )

    print("\n=== BEST RAW OUTPUT ===")
    print(best.out[-2500:])

    if best.speedup >= args.target_speedup:
        print(f"\nPASS: best speedup {best.speedup:.2f}x >= target {args.target_speedup:.2f}x")
        return 0
    print(f"\nBELOW TARGET: best speedup {best.speedup:.2f}x < target {args.target_speedup:.2f}x")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

