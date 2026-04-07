#!/usr/bin/env python3
"""
Smoke test + tiny demo for Brain early-exit (PyTorch only — not in llama-server GGUF).

  cd train && python ../scripts/demo_early_exit.py

After training with default early_exit_loss_weight, compare:

  generate_brain_only vs generate_brain_early_exit

on the same prompt; verbose prints shallow-step fraction.
"""
from __future__ import annotations

import os
import sys

# train/ as cwd for imports
_TRAIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train")
sys.path.insert(0, _TRAIN)

import torch
from config import BrainConfig, DraftConfig
from model import SpinalCordLLM


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bc = BrainConfig()
    # Match small train_brain defaults enough to run on CPU/GPU
    bc.n_layers = 8
    bc.d_model = 512
    bc.d_ff = 1408
    bc.n_heads = 8
    bc.n_kv_heads = 4
    bc.max_seq_len = 64
    bc.early_exit_after = 4
    bc.early_exit_loss_weight = 0.25

    draft = DraftConfig()
    draft.n_layers = 4
    draft.d_model = 384
    draft.d_ff = 1024
    draft.n_heads = 6
    draft.n_kv_heads = 3

    model = SpinalCordLLM(draft, bc)
    model.brain.to(device)

    B, T = 1, 12
    x = torch.randint(0, min(1000, bc.vocab_size), (B, T), device=device)

    logits, le = model.brain(x, return_early_logits=True)
    assert logits.shape == (B, T, bc.vocab_size)
    assert le.shape == logits.shape
    print("forward + early logits OK:", logits.shape)

    # Short gen (untrained — expect mostly full-depth steps)
    start = torch.randint(0, min(1000, bc.vocab_size), (1, 6), device=device)
    out = model.generate_brain_early_exit(
        start.cpu(),
        max_new_tokens=8,
        brain_device=device,
        verbose=True,
        temperature=0.8,
        early_exit_max_prob=0.99,
    )
    print("generated length:", out.shape[1])


if __name__ == "__main__":
    main()
