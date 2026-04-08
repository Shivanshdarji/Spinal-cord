#!/usr/bin/env python3
"""
Hackathon baseline entrypoint (required at repository root).

This script runs SpinalCord Bench tasks through:
1) OpenEnv server (Space/local) via WebSocket client
2) OpenAI-compatible model endpoint (SpinalCord llama-server style)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parent
BENCH_DIR = REPO_ROOT / "envs" / "spinalcord_bench"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from spinalcord_bench.client import SpinalBenchClient  # noqa: E402
from spinalcord_bench.models import SpinalBenchAction, SpinalBenchObservation  # noqa: E402
from spinalcord_bench.server.graders import TASK_ORDER  # noqa: E402


def _prompt(obs: SpinalBenchObservation) -> str:
    return f"{obs.instruction}\n\nContext:\n{obs.reference_context}"


def run_episode(env, llm: OpenAI, model: str, task_id: str, seed: int) -> tuple[float, str]:
    result = env.reset(seed=seed, task_id=task_id)
    obs = result.observation
    messages = [
        {
            "role": "system",
            "content": "Answer precisely using the provided context only.",
        },
        {"role": "user", "content": _prompt(obs)},
    ]

    done = False
    final_score = 0.0
    detail = ""
    while not done:
        rsp = llm.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=128,
            temperature=0.2,
        )
        answer = (rsp.choices[0].message.content or "").strip() or " "
        step = env.step(SpinalBenchAction(content=answer))
        obs = step.observation
        done = step.done
        final_score = obs.grader_score
        detail = obs.grader_detail
        messages.append({"role": "assistant", "content": answer})
    return final_score, detail


def main() -> int:
    openenv_base = os.environ.get("OPENENV_BASE_URL", "http://127.0.0.1:7860")
    openai_base = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN") or "unused"
    model = os.environ.get("OPENAI_MODEL", "")

    if not model:
        print("Missing OPENAI_MODEL (set to model id from /v1/models).", file=sys.stderr)
        return 2

    llm = OpenAI(base_url=openai_base, api_key=api_key)

    scores: list[float] = []
    with SpinalBenchClient(base_url=openenv_base).sync() as env:
        for i, task_id in enumerate(TASK_ORDER):
            score, detail = run_episode(env, llm, model, task_id, seed=100 + i)
            scores.append(score)
            print(f"{task_id}: score={score:.3f} detail={detail}")

    avg = sum(scores) / len(scores)
    print(f"average_score={avg:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
