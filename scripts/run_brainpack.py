#!/usr/bin/env python3
"""
Run the pluggable SpinalCord engine using a named BrainPack profile.

Example:
  python scripts/run_brainpack.py --pack spinalcord_custom --prompt "hello"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any
from urllib import request, error

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "train"))

from config import BrainConfig, SpinalCordConfig  # type: ignore
from model import SpinalCordBrain, SpinalCordDraft  # type: ignore
from pluggable_spinalcord import (  # type: ignore
    SpinalCordEngine,
    SpinalCordRuntimeConfig,
    TorchBrainAdapter,
    TorchDraftAdapter,
)
from tokenizer_sc import load_tokenizer_and_export  # type: ignore


def _load_brainpacks(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "packs" not in data:
        raise ValueError(f"Invalid brainpacks file: {path}")
    return data


def _require_file(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)


def run_pytorch_spinalcord_pack(
    pack_name: str,
    pack: dict[str, Any],
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gamma = int(pack.get("gamma", 8))
    acceptance_floor = float(pack.get("acceptance_floor", 0.0))
    expected_vocab_size = int(pack.get("tokenizer_vocab_size", 32000))

    brain_ckpt = os.path.join(_ROOT, str(pack["brain_ckpt"]))
    draft_ckpt = os.path.join(_ROOT, str(pack["draft_ckpt"]))
    _require_file(brain_ckpt)
    _require_file(draft_ckpt)

    cfg = SpinalCordConfig()
    bundle, _ = load_tokenizer_and_export(expected_vocab_size=expected_vocab_size or cfg.brain.vocab_size)

    brain_state = torch.load(brain_ckpt, map_location=device, weights_only=False)
    draft_state = torch.load(draft_ckpt, map_location=device, weights_only=False)
    brain_cfg: BrainConfig = brain_state["cfg"]
    draft_cfg = draft_state["cfg"]

    brain = SpinalCordBrain(brain_cfg).to(device)
    draft = SpinalCordDraft(draft_cfg).to(device)
    brain.load_state_dict(brain_state["model_state"])
    draft.load_state_dict(draft_state["model_state"])
    brain.eval()
    draft.eval()

    engine = SpinalCordEngine(
        brain=TorchBrainAdapter(brain, vocab_size=brain_cfg.vocab_size),
        draft=TorchDraftAdapter(draft, vocab_size=draft_cfg.vocab_size),
        cfg=SpinalCordRuntimeConfig(
            gamma=gamma if gamma > 0 else int(draft_cfg.gamma),
            acceptance_floor=acceptance_floor,
        ),
    )

    input_ids = torch.tensor([bundle.encode(prompt)], dtype=torch.long, device=device)
    out = engine.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        brain_device=device,
        draft_device=device,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        verbose=True,
    )
    gen = out[0].tolist()[len(input_ids[0]) :]
    text = bundle.decode(gen)

    print(f"\n=== BrainPack: {pack_name} ===")
    print(text[:2000])
    return 0


def _http_json(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = request.Request(url=url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=120) as r:
            raw = r.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {url}: {body[:1000]}") from e
    except Exception as e:
        raise RuntimeError(f"{method} {url} failed: {e}") from e


def _pick_model_id_from_models(j: dict[str, Any]) -> str | None:
    raw = []
    if isinstance(j.get("data"), list):
        raw = j["data"]
    elif isinstance(j.get("models"), list):
        raw = j["models"]
    ids: list[str] = []
    for m in raw:
        if isinstance(m, dict):
            mid = m.get("id") or m.get("name") or m.get("model")
            if mid is not None:
                ids.append(str(mid))
    if not ids:
        return None
    for mid in ids:
        low = mid.lower()
        if "brain" in low and "draft" not in low:
            return mid
    for mid in ids:
        if "draft" not in mid.lower():
            return mid
    return ids[0]


def run_llama_server_chat_pack(
    pack_name: str,
    pack: dict[str, Any],
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> int:
    endpoint = str(pack.get("endpoint", "http://127.0.0.1:8080")).rstrip("/")
    model_id = str(pack.get("model_id", "")).strip()
    auto_discover = bool(pack.get("auto_discover_model_id", True))
    repeat_penalty = float(pack.get("repeat_penalty", 1.15))

    if auto_discover or not model_id:
        models_json = _http_json("GET", endpoint + "/v1/models")
        auto_id = _pick_model_id_from_models(models_json)
        if auto_id:
            model_id = auto_id
    if not model_id:
        raise RuntimeError(
            f"Could not resolve model id from {endpoint}/v1/models. "
            "Set model_id explicitly in configs/brainpacks.json."
        )

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_new_tokens,
        "repeat_penalty": repeat_penalty,
    }
    resp = _http_json("POST", endpoint + "/v1/chat/completions", payload)

    content = ""
    try:
        content = str(resp["choices"][0]["message"]["content"])
    except Exception:
        content = json.dumps(resp)[:2000]

    print(f"\n=== BrainPack: {pack_name} (llama-server chat) ===")
    print(f"endpoint: {endpoint}")
    print(f"model:    {model_id}")
    print(content[:2000])
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Run pluggable SpinalCord with a BrainPack profile.")
    p.add_argument("--pack", type=str, default="", help="BrainPack name in configs/brainpacks.json")
    p.add_argument("--brainpacks", type=str, default=os.path.join("configs", "brainpacks.json"))
    p.add_argument("--prompt", type=str, default="Explain recursion in 3 short bullet points.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)
    args = p.parse_args()

    bp_path = os.path.join(_ROOT, args.brainpacks)
    if not os.path.isfile(bp_path):
        print(f"BrainPacks config not found: {bp_path}", file=sys.stderr)
        return 2

    all_data = _load_brainpacks(bp_path)
    packs = all_data["packs"]
    pack_name = args.pack.strip() or str(all_data.get("default_pack", "")).strip()
    if not pack_name:
        print("No --pack provided and no default_pack in config.", file=sys.stderr)
        return 2
    if pack_name not in packs:
        print(f"Unknown pack '{pack_name}'. Available: {', '.join(sorted(packs.keys()))}", file=sys.stderr)
        return 2

    pack = packs[pack_name]
    kind = str(pack.get("kind", "")).strip()

    if kind == "pytorch_spinalcord_pt":
        return run_pytorch_spinalcord_pack(
            pack_name=pack_name,
            pack=pack,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

    if kind == "external_adapter_stub":
        print(
            f"Pack '{pack_name}' is a stub profile for future adapter wiring.\n"
            f"Notes: {pack.get('notes', '(no notes)')}\n"
            "Today, use dashboard/run_dashboard_llama_scaffold.bat for llama scaffold chat."
        )
        return 0
    if kind == "llama_server_chat":
        return run_llama_server_chat_pack(
            pack_name=pack_name,
            pack=pack,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    print(f"Unsupported pack kind: {kind}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
