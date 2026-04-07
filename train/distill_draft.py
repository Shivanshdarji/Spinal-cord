# pyre-ignore-all-errors
"""
SpinalCord LLM — Draft Distillation (SCDraft-120M)
====================================================
This is the KEY to 10x speed:
  Instead of training the Draft on raw text (like the Brain),
  we train it to COPY the Brain's probability distributions.

  When Draft predicts token "Paris" with 92% probability
  and Brain also gives "Paris" 94% → ACCEPTED.
  This is knowledge distillation — the Draft becomes a
  "compressed echo" of the Brain.

  Target acceptance rate: 85-90% → ~7-8x real speedup.

Algorithm:
  For each text batch:
    1. Run Brain (frozen) → get soft probability distribution (teacher)
    2. Run Draft (training) → get its distribution (student)
    3. Loss = alpha * KL_div(Draft || Brain) + (1-alpha) * CrossEntropy
    4. Backprop into Draft only — Brain stays frozen!

Run:
    python distill_draft.py --brain_ckpt ../models/scbrain_best.pt

Author: Shivansh Darji | AppDice
"""

import os
import sys
import math
import time
import argparse
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from torch.cuda.amp import GradScaler, autocast # type: ignore
from torch.utils.data import DataLoader, IterableDataset # type: ignore
from typing import List, Tuple, Any, Optional, Iterator, Callable

sys.path.insert(0, os.path.dirname(__file__))
from config import BrainConfig, DraftConfig, DistillConfig
from model import SpinalCordBrain, SpinalCordDraft
from tokenizer_sc import load_tokenizer_and_export
from dataset import MixedGeneralInstructionDataset, MixedConversationDataset


class TextDataset(IterableDataset):
    def __init__(self, encode_fn, seq_len=1024):
        self.encode   = encode_fn
        self.seq_len  = seq_len

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        from datasets import load_dataset # type: ignore
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        chunk_size = self.seq_len + 1
        buffer: List[int] = []
        for sample in ds:
            tokens: List[int] = self.encode(sample.get("text", "") + " ")
            buffer.extend(tokens) # type: ignore
            while len(buffer) >= chunk_size:
                chunk = buffer[:chunk_size] # type: ignore
                buffer = buffer[self.seq_len:] # type: ignore
                x = torch.tensor(chunk[:-1], dtype=torch.long) # type: ignore
                y = torch.tensor(chunk[1:],  dtype=torch.long) # type: ignore
                yield x, y


# ─── KL Divergence Distillation Loss ─────────────────────────────────────────
def distillation_loss(
    draft_logits:  torch.Tensor,   # [B, S, V] — student
    brain_logits:  torch.Tensor,   # [B, S, V] — teacher (frozen)
    labels:        torch.Tensor,   # [B, S]
    temperature:   float = 4.0,
    alpha:         float = 0.9,
) -> torch.Tensor:
    """
    Combined KL-divergence distillation + cross-entropy loss.

    KL divergence pulls the Draft distribution toward the Brain's.
    Higher temperature 'softens' the Brain's distribution so the
    Draft can learn from all the probability mass, not just the top-1.
    """
    B, S, V = draft_logits.shape

    # 1. Soft targets (KL loss at high temperature)
    with torch.no_grad():
        brain_probs = F.softmax(brain_logits / temperature, dim=-1)

    draft_log_probs = F.log_softmax(draft_logits / temperature, dim=-1)

    # KL(student || teacher) — shape [B*S]
    kl_loss = F.kl_div(
        draft_log_probs.view(B * S, V),
        brain_probs.view(B * S, V),
        reduction="batchmean"
    ) * (temperature ** 2)   # rescale by T² as per Hinton et al.

    # 2. Hard CE loss (keeps Draft grounded in real labels)
    ce_loss = F.cross_entropy(
        draft_logits.view(B * S, V),
        labels.view(B * S),
        ignore_index=-1,
    )

    # 3. Combined
    return alpha * kl_loss + (1.0 - alpha) * ce_loss


# ─── Main Distillation Loop ───────────────────────────────────────────────────
def distill(args):
    dcfg   = DraftConfig()
    bcfg   = BrainConfig()
    distcfg = DistillConfig()

    # Override steps if passed via CLI
    if getattr(args, "steps", None):
        distcfg.max_steps = args.steps

    # VRAM-safe distillation settings for RTX 2050 (4GB)
    # Keep this aligned with Brain training context length to avoid OOM.
    distcfg.max_seq_len = 384
    distcfg.batch_size  = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print("  SCDraft-120M Distillation | AppDice | Shivansh Darji")
    print(f"  Device:      {device.upper()}")
    print(f"  Temperature: {distcfg.temperature} (softens Brain's targets)")
    print(f"  Alpha:       {distcfg.alpha} ({distcfg.alpha*100:.0f}% KL + {(1-distcfg.alpha)*100:.0f}% CE)")
    print(f"  Steps:       {distcfg.max_steps:,}")
    print(f"{'='*60}\n")

    brain_ckpt = args.brain_ckpt or os.path.join("..", "models", "scbrain_best.pt")
    if os.path.exists(brain_ckpt):
        ckpt = torch.load(brain_ckpt, map_location=device, weights_only=False)
        # Prefer the exact config used during Brain training.
        # train_brain.py saves it under "cfg".
        bcfg = ckpt.get("cfg", ckpt.get("config", bcfg))
        brain = SpinalCordBrain(bcfg).to(device)
        brain.load_state_dict(ckpt["model_state"])
        print(f"[Brain] Loaded checkpoint: {brain_ckpt}")
    else:
        brain = SpinalCordBrain(bcfg).to(device)
        print(f"[Brain] WARNING: No checkpoint found at {brain_ckpt}")
        print(f"        Using random Brain weights for demo (train Brain first!)")

    brain.eval()
    for p in brain.parameters():
        p.requires_grad = False   # 🔒 Brain is FROZEN
    print(f"[Brain] Frozen. Total params: {sum(p.numel() for p in brain.parameters())/1e6:.0f}M")

    # ── Initialize Draft (student) ───────────────────────────────────────────
    draft = SpinalCordDraft(dcfg).to(device)   # Train Draft on same device as Brain
    print(f"[Draft] Student params: {sum(p.numel() for p in draft.parameters())/1e6:.0f}M")

    # Tokenizer
    bundle, export_dir = load_tokenizer_and_export(
        expected_vocab_size=dcfg.vocab_size,
    )
    if bcfg.vocab_size != dcfg.vocab_size:
        raise RuntimeError(f"Brain/Draft vocab mismatch: brain={bcfg.vocab_size} draft={dcfg.vocab_size}")
    encode = bundle.encode
    print(f"[Tokenizer] Exported tokenizer files -> {export_dir}")

    # Dataset + DataLoader
    if args.data_mode == "mixed":
        print(f"[Dataset] Using mixed general+instruction (instruction_ratio={args.instruction_ratio:.2f})...")
        dataset = MixedGeneralInstructionDataset(
            encode_fn=encode,
            seq_len=distcfg.max_seq_len,
            instruction_ratio=args.instruction_ratio,
            max_tokens_per_example=args.max_tokens_per_example,
            max_segments_per_example=args.max_segments_per_example,
        )
    elif args.data_mode == "conversation":
        print(
            f"[Dataset] Conversation-first mix: TinyStories + UltraChat + Alpaca "
            f"(story={args.conv_story:.2f}, dialog={args.conv_dialog:.2f}, inst={args.conv_inst:.2f})..."
        )
        dataset = MixedConversationDataset(
            encode_fn=encode,
            seq_len=distcfg.max_seq_len,
            story_ratio=args.conv_story,
            dialogue_ratio=args.conv_dialog,
            instruction_ratio=args.conv_inst,
            max_tokens_per_example=args.max_tokens_per_example,
            max_segments_per_example=args.max_segments_per_example,
            chat_split=args.conv_chat_split,
        )
    else:
        print("[Dataset] Using TinyStories only...")
        dataset = TextDataset(encode, seq_len=distcfg.max_seq_len)
    loader = DataLoader(dataset, batch_size=distcfg.batch_size, num_workers=0)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        draft.parameters(),
        lr=distcfg.learning_rate,
        weight_decay=0.01
    )

    def lr_lambda(step):
        if step < distcfg.warmup_steps:
            return step / distcfg.warmup_steps
        decay = (step - distcfg.warmup_steps) / (distcfg.max_steps - distcfg.warmup_steps)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * decay)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # Distillation uses bf16 autocast on CUDA for stability + lower memory.
    use_amp = (device == "cuda")
    if use_amp:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        except Exception:
            pass
    scaler = GradScaler(enabled=False)

    ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_loss   = float("inf")
    window_accept = 0.0
    window_tokens = 0
    overall_accept = 0.0
    overall_tokens = 0

    draft.train()
    step = 0
    t0   = time.time()

    print("\n[Distill] Starting knowledge distillation...\n")
    for batch_x, batch_y in loader:
        if step >= distcfg.max_steps:
            break

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        try:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):  # type: ignore[attr-defined]
                # ── Get Brain's soft targets (frozen) ─────────────────────────
                with torch.no_grad():
                    brain_logits = brain(batch_x)

                # ── Draft forward pass (same device) ──────────────────────────
                draft_logits = draft(batch_x)

                # ── Distillation loss ─────────────────────────────────────────
                # Compute loss in fp32 for numerical stability.
                loss = distillation_loss(
                    draft_logits.float(),
                    brain_logits.float(),
                    batch_y,
                    temperature=distcfg.temperature,
                    alpha=distcfg.alpha,
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(draft.parameters(), distcfg.grad_clip)
            optimizer.step()
            scheduler.step()
        except RuntimeError as e:
            msg = str(e).lower()
            if device == "cuda" and ("out of memory" in msg or "cublas_status_execution_failed" in msg):
                print("[Distill] CUDA matmul/OOM error; clearing cache and skipping batch...")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            raise

        # ── Estimate acceptance rate (simulation) ─────────────────────────
        with torch.no_grad():
            draft_probs = F.softmax(draft_logits[:, :-1, :].float(), dim=-1)
            brain_probs = F.softmax(brain_logits[:, :-1, :].float(), dim=-1)
            top_ids = brain_probs.argmax(dim=-1)
            dp = draft_probs.gather(-1, top_ids.unsqueeze(-1)).squeeze(-1)
            bp = brain_probs.gather(-1, top_ids.unsqueeze(-1)).squeeze(-1)
            ratios = torch.minimum(torch.ones_like(dp), bp / (dp + 1e-9))
            acc = float(ratios.sum().item())
            n = int(ratios.numel())
            window_accept += acc
            window_tokens += n
            overall_accept += acc
            overall_tokens += n

        step += 1

        if step % 50 == 0:
            elapsed     = time.time() - t0
            accept_rate = window_accept / max(window_tokens, 1) * 100
            lr_now      = scheduler.get_last_lr()[0]
            speedup_est = dcfg.gamma * (accept_rate / 100)
            print(f"Step {step:6d}/{distcfg.max_steps} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Accept: {accept_rate:.1f}% | "
                  f"Est speedup: {speedup_est:.1f}x | "
                  f"LR: {lr_now:.2e}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                ckpt_path = os.path.join(ckpt_dir, "scdraft_best.pt")
                torch.save({
                    "step": step,
                    "loss": best_loss,
                    "model_state": draft.state_dict(),
                    "cfg": dcfg,
                }, ckpt_path)
                print(f"  Saved best draft checkpoint -> {ckpt_path}")

            window_accept = 0.0
            window_tokens = 0

    print(f"\n{'='*60}")
    accept_rate = overall_accept / max(overall_tokens, 1) * 100
    print(f"  Distillation complete!")
    print(f"  Best distillation loss: {best_loss:.4f}")
    print(f"  Final acceptance rate:  {accept_rate:.1f}%")
    speedup = dcfg.gamma * (accept_rate / 100)
    print(f"  Estimated SpinalCord speedup: {speedup:.1f}x")
    print(f"  Next step: python convert/convert_both.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--brain_ckpt", type=str, default=None,
                        help="Path to trained Brain checkpoint")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override max distillation steps")
    parser.add_argument(
        "--data_mode",
        type=str,
        default="mixed",
        choices=["tinystories", "mixed", "conversation"],
        help="Distillation data mode (conversation = story + dialogue + instruction).",
    )
    parser.add_argument(
        "--instruction_ratio",
        type=float,
        default=0.5,
        help="Probability that the next streamed example comes from instruction/Q&A data (mixed mode).",
    )
    parser.add_argument(
        "--max_tokens_per_example",
        type=int,
        default=4096,
        help="Truncate tokenized examples to this max token length (mixed mode).",
    )
    parser.add_argument(
        "--max_segments_per_example",
        type=int,
        default=8,
        help="Cap how many seq_len segments we emit per streamed example (mixed mode).",
    )
    parser.add_argument(
        "--conv_story",
        type=float,
        default=0.35,
        help="[conversation] Relative weight for TinyStories.",
    )
    parser.add_argument(
        "--conv_dialog",
        type=float,
        default=0.35,
        help="[conversation] Relative weight for UltraChat (multi-turn).",
    )
    parser.add_argument(
        "--conv_inst",
        type=float,
        default=0.30,
        help="[conversation] Relative weight for Alpaca instruction data.",
    )
    parser.add_argument(
        "--conv_chat_split",
        type=str,
        default="train_sft",
        help="[conversation] HuggingFace split for UltraChat (train_sft, train_gen, ...).",
    )
    args = parser.parse_args()
    distill(args)
