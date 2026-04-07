#!/usr/bin/env python3
"""
Demo: use the generic SpinalCordEngine with pluggable adapters.

Today this wraps the existing local SpinalCord Brain+Draft checkpoints, but the
engine interface is designed so you can swap in other Brain/Draft adapters
(e.g. llama.cpp-backed adapters later) without touching acceptance logic.
"""

from __future__ import annotations

import os
import sys

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "train"))

from config import SpinalCordConfig, BrainConfig  # type: ignore
from model import SpinalCordBrain, SpinalCordDraft  # type: ignore
from pluggable_spinalcord import (  # type: ignore
    SpinalCordEngine,
    SpinalCordRuntimeConfig,
    TorchBrainAdapter,
    TorchDraftAdapter,
)
from tokenizer_sc import load_tokenizer_and_export  # type: ignore


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SpinalCordConfig()

    bundle, _ = load_tokenizer_and_export(expected_vocab_size=cfg.brain.vocab_size)

    brain_ckpt = os.path.join(_ROOT, "models", "scbrain_best.pt")
    draft_ckpt = os.path.join(_ROOT, "models", "scdraft_best.pt")

    if not os.path.isfile(brain_ckpt) or not os.path.isfile(draft_ckpt):
        raise FileNotFoundError(
            "Missing checkpoints: expected models/scbrain_best.pt and models/scdraft_best.pt"
        )

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
        cfg=SpinalCordRuntimeConfig(gamma=int(draft_cfg.gamma), acceptance_floor=0.0),
    )

    prompt = "Explain recursion in 3 short bullet points."
    input_ids = torch.tensor([bundle.encode(prompt)], dtype=torch.long, device=device)

    out = engine.generate(
        input_ids,
        max_new_tokens=80,
        brain_device=device,
        draft_device=device,
        temperature=0.2,
        top_k=40,
        top_p=0.95,
        verbose=True,
    )

    gen = out[0].tolist()[len(input_ids[0]) :]
    text = bundle.decode(gen)

    print("\n=== Pluggable SpinalCordEngine output ===")
    print(text[:1200])


if __name__ == "__main__":
    main()
