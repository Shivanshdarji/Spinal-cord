#!/usr/bin/env python3
"""
Run all three tasks against an OpenAI-compatible model (SpinalCord via llama-server).

Environment variables
---------------------
OPENENV_BASE_URL   WebSocket/HTTP base for the OpenEnv server (default http://127.0.0.1:7860)
OPENAI_BASE_URL    OpenAI-compatible chat endpoint (default http://127.0.0.1:8080/v1)
OPENAI_API_KEY     API key; if unset, HF_TOKEN is used for providers that expect a Bearer token
HF_TOKEN           Hugging Face token (optional; used when OPENAI_API_KEY is empty)
OPENAI_MODEL       Model id as seen by the chat server (default: local placeholder — set to your Brain id)
"""

from __future__ import annotations

import os
import sys

from openai import OpenAI

from spinalcord_bench.client import SpinalBenchClient
from spinalcord_bench.models import SpinalBenchObservation, SpinalBenchAction
from spinalcord_bench.server.graders import TASK_ORDER


def _user_prompt(obs: SpinalBenchObservation) -> str:
    return f"{obs.instruction}\n\nContext:\n{obs.reference_context}"


def run_episode(
    env,
    llm: OpenAI,
    model: str,
    task_id: str,
    seed: int,
) -> tuple[float, str, bool]:
    r = env.reset(seed=seed, task_id=task_id)
    obs = r.observation
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise assistant. Follow instructions exactly. "
                "Give the shortest correct answer."
            ),
        },
        {"role": "user", "content": _user_prompt(obs)},
    ]
    done = False
    final_score = 0.0
    detail = ""
    while not done:
        rsp = llm.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=512,
            temperature=0.2,
        )
        text = (rsp.choices[0].message.content or "").strip()
        if not text:
            text = " "
        step = env.step(SpinalBenchAction(content=text))
        obs = step.observation
        final_score = obs.grader_score
        detail = obs.grader_detail
        done = step.done
        messages.append({"role": "assistant", "content": text})
    return final_score, detail, obs.step_limit_reached


def main() -> int:
    base = os.environ.get("OPENENV_BASE_URL", "http://127.0.0.1:7860")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN") or "unused"
    llm = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"),
        api_key=api_key,
    )
    model = os.environ.get("OPENAI_MODEL", "")
    if not model:
        print(
            "Set OPENAI_MODEL to your served model id (see llama-server /v1/models).",
            file=sys.stderr,
        )
        return 2

    rows: list[tuple[str, float, str, str]] = []
    with SpinalBenchClient(base_url=base).sync() as env:
        for i, task_id in enumerate(TASK_ORDER):
            score, detail, limited = run_episode(env, llm, model, task_id, seed=100 + i)
            status = "ok" if score >= 0.99 else ("partial" if score > 0 else "fail")
            if limited:
                status = "limit"
            rows.append((task_id, score, detail, status))

    print("task_id | grader_score | detail | status")
    for task_id, score, detail, status in rows:
        print(f"{task_id} | {score:.3f} | {detail} | {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
