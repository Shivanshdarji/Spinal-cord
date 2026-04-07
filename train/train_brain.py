# pyre-ignore-all-errors
"""
SpinalCord LLM - Brain Training (SCBrain-400M)
================================================
Trains a 400M parameter Brain model tuned to fit inside 4GB VRAM
(RTX 2050) with gradient accumulation for effective large batches.

Architecture: LLaMA-style (RoPE, RMSNorm, SwiGLU, GQA)
Dataset:      TinyStories (streaming)
Precision:    bfloat16 mixed precision

Run:
    python train_brain.py              # full 6000 step training
    python train_brain.py --steps 100  # quick smoke test

Author: Shivansh Darji | AppDice
"""

import os
import sys
import math
import time
import argparse

# Helps reduce CUDA memory fragmentation on long runs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch # type: ignore
import torch.amp # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from torch.utils.data import DataLoader, IterableDataset # type: ignore
from typing import List, Tuple, Any, Optional, Iterator, Callable

# Add train/ dir to path so we can import config and model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BrainConfig
from model import SpinalCordBrain
from tokenizer_sc import load_tokenizer_and_export
from dataset import MixedGeneralInstructionDataset, MixedConversationDataset


# ─── Streaming Dataset ────────────────────────────────────────────────────────
class TinyStoriesDataset(IterableDataset):
    """Streams TinyStories from HuggingFace, packs tokens into fixed chunks."""

    def __init__(self, encode_fn, seq_len: int, split: str = "train"):
        self.encode  = encode_fn
        self.seq_len = seq_len
        self.split   = split

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        from datasets import load_dataset # type: ignore
        print(f"[Dataset] Streaming TinyStories/{self.split}...")
        ds = load_dataset(
            "roneneldan/TinyStories",
            split=self.split,
            streaming=True,
        )
        chunk_size = self.seq_len + 1
        buffer: List[int] = []
        for sample in ds:
            text = str(sample.get("text", "")) + " "
            tokens: List[int] = self.encode(text)
            buffer.extend(tokens) # type: ignore
            while len(buffer) >= chunk_size:
                chunk  = buffer[:chunk_size] # type: ignore
                buffer = buffer[self.seq_len :] # type: ignore
                x = torch.tensor(chunk[:-1], dtype=torch.long) # type: ignore
                y = torch.tensor(chunk[1:],  dtype=torch.long) # type: ignore
                yield x, y


# ─── LR Schedule ──────────────────────────────────────────────────────────────
def get_lr(step: int, warmup: int, total: int, peak_lr: float, min_lr: float = 1e-5) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup:
        return peak_lr * step / max(warmup, 1)
    if step >= total:
        return min_lr
    decay = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * decay))


# ─── Training ─────────────────────────────────────────────────────────────────
def train(args):
    cfg = BrainConfig()

    # Override steps if passed via CLI
    if args.steps:
        cfg.max_steps = args.steps
    if args.max_seq_len is not None:
        cfg.max_seq_len = int(args.max_seq_len)
    if args.grad_accum is not None:
        cfg.grad_accum_steps = int(args.grad_accum)
    if args.early_exit_after is not None:
        cfg.early_exit_after = int(args.early_exit_after)
    if args.early_exit_loss_weight is not None:
        cfg.early_exit_loss_weight = float(args.early_exit_loss_weight)

    # VRAM-safe settings for RTX 2050 (4GB)
    # NOTE: PyTorch + CUDA kernels can add overhead; keep this conservative.
    cfg.n_layers         = 10      # smaller = less activation memory
    cfg.d_model          = 1024    # smaller hidden size = big VRAM win
    cfg.d_ff             = 2816    # ~2.75x d_model, keeps SwiGLU reasonable
    cfg.n_heads          = 8
    cfg.n_kv_heads       = 4
    cfg.max_seq_len      = 384     # shorter context = much less VRAM
    cfg.batch_size       = 1       # per-GPU micro batch
    cfg.grad_accum_steps = 16      # effective batch = 16
    cfg.use_bf16        = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]

    vram_used = "~2.5 GB" if device == "cuda" else "N/A"
    param_est = (12 * (1536 * (12 * 128 + 2 * 4 * 128 + 1536) +
                       1536 * 4096 * 3) + 32000 * 1536) / 1e6

    print(f"\n{'='*60}")
    print(f"  SCBrain Training | AppDice | Shivansh Darji")
    print(f"  Device:  {device.upper()}")
    print(
        f"  Layers:  {cfg.n_layers} | d_model: {cfg.d_model} | heads: {cfg.n_heads} | "
        f"early_exit@{getattr(cfg, 'early_exit_after', 0)} w={getattr(cfg, 'early_exit_loss_weight', 0)}"
    )
    print(f"  Seq len: {cfg.max_seq_len} | Batch: {cfg.batch_size} x {cfg.grad_accum_steps} accum")
    print(f"  Est params: ~{param_est:.0f}M | VRAM: {vram_used}")
    print(f"  Steps:   {cfg.max_steps:,}")
    print(f"{'='*60}\n")

    # Build model
    model = SpinalCordBrain(cfg).to(device)
    actual_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Brain] Actual params: {actual_params:.1f}M\n")

    # Tokenizer (shared across Brain + Draft) + export for GGUF conversion
    bundle, export_dir = load_tokenizer_and_export(
        expected_vocab_size=cfg.vocab_size,
    )
    encode = bundle.encode
    print(f"[Tokenizer] Exported tokenizer files -> {export_dir}")
    if args.data_mode == "mixed":
        print(f"[Dataset] Using mixed general+instruction (instruction_ratio={args.instruction_ratio:.2f})...")
        dataset = MixedGeneralInstructionDataset(
            encode_fn=encode,
            seq_len=cfg.max_seq_len,
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
            seq_len=cfg.max_seq_len,
            story_ratio=args.conv_story,
            dialogue_ratio=args.conv_dialog,
            instruction_ratio=args.conv_inst,
            max_tokens_per_example=args.max_tokens_per_example,
            max_segments_per_example=args.max_segments_per_example,
            chat_split=args.conv_chat_split,
        )
    else:
        print("[Dataset] Using TinyStories only...")
        dataset = TinyStoriesDataset(encode, seq_len=cfg.max_seq_len)
    loader  = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=0)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
        fused=(device == "cuda"),   # fused Adam is faster on CUDA
    )

    # Mixed precision scaler
    use_amp = (device == "cuda" and cfg.use_bf16)
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp) # type: ignore

    ckpt_dir  = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_loss = float("inf")
    ckpt_path = os.path.join(ckpt_dir, "scbrain_best.pt")

    # Warm-start from existing checkpoint if present (weights-only resume)
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            inc = model.load_state_dict(ckpt["model_state"], strict=False)
            if inc.missing_keys:
                print(f"[Brain] Warm-start missing keys (ok for new early-exit layers): {len(inc.missing_keys)}")
            if inc.unexpected_keys:
                print(f"[Brain] Warm-start unexpected keys: {inc.unexpected_keys[:8]}...")
            best_loss = float(ckpt.get("loss", best_loss))
            print(f"[Brain] Warm-started weights from: {ckpt_path} (prev best loss={best_loss:.4f})")
        except Exception as e:
            print(f"[Brain] WARNING: Could not warm-start from {ckpt_path}: {e}")

    step       = 0
    loss_accum = 0.0
    t0         = time.time()
    t_log      = t0

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for batch_x, batch_y in loader:
        if step >= cfg.max_steps:
            break

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Forward pass with optional bfloat16
        amp_dtype = torch.bfloat16 if use_amp else torch.float32
        try:
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp): # type: ignore
                ew = float(getattr(cfg, "early_exit_loss_weight", 0.0) or 0.0)
                if ew > 0 and getattr(model, "_early_enabled", False):
                    logits, logits_e = model(batch_x, return_early_logits=True)
                    loss_main = F.cross_entropy(
                        logits.reshape(-1, cfg.vocab_size),
                        batch_y.reshape(-1),
                        ignore_index=-1,
                    )
                    loss_early = F.cross_entropy(
                        logits_e.reshape(-1, cfg.vocab_size),
                        batch_y.reshape(-1),
                        ignore_index=-1,
                    )
                    loss = (loss_main + ew * loss_early) / cfg.grad_accum_steps
                else:
                    logits = model(batch_x)
                    loss = F.cross_entropy(
                        logits.reshape(-1, cfg.vocab_size),
                        batch_y.reshape(-1),
                        ignore_index=-1,
                    ) / cfg.grad_accum_steps

            scaler.scale(loss).backward()
        except RuntimeError as e:
            if device == "cuda" and "out of memory" in str(e).lower():
                print(
                    "[Brain] CUDA OOM; skipping batch. Prefer: --max-seq-len 256 "
                    "(and/or --grad-accum 8). Stopping — fix VRAM and re-run."
                )
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    except RuntimeError:
                        pass
                raise SystemExit(1) from e
            raise
        loss_accum += loss.item()

        # Only update weights every grad_accum_steps
        if (step + 1) % cfg.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            lr = get_lr(step, cfg.warmup_steps, cfg.max_steps, cfg.learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Reduce fragmentation during long runs
            if device == "cuda" and (step + 1) % (cfg.grad_accum_steps * 25) == 0:
                torch.cuda.empty_cache()

            # Logging
            smooth_loss = loss_accum * cfg.grad_accum_steps
            elapsed     = time.time() - t0
            tok_s       = (step + 1) * cfg.batch_size * cfg.max_seq_len / elapsed

            print(
                f"Step {step+1:6d}/{cfg.max_steps} | "
                f"Loss: {smooth_loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Tok/s: {tok_s:,.0f}"
            )

            # Save best
            if smooth_loss < best_loss:
                best_loss = smooth_loss
                torch.save({
                    "step":        step,
                    "loss":        best_loss,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "cfg":         cfg,
                }, ckpt_path)
                print(f"  Saved best checkpoint -> {ckpt_path}")

            loss_accum = 0.0

        step += 1

    print(f"\n{'='*60}")
    print(f"  Training done!  Best loss: {best_loss:.4f}")
    print(f"  Checkpoint   :  models/scbrain_best.pt")
    print(f"  Next step    :  python distill_draft.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SCBrain LLM")
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Number of training steps (default: from config)"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Override max sequence length (default 384). Use 256 on 4GB GPU if you hit CUDA OOM.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Override gradient accumulation steps (default 16). Smaller can lower peak VRAM at optimizer step.",
    )
    parser.add_argument(
        "--data_mode",
        type=str,
        default="mixed",
        choices=["tinystories", "mixed", "conversation"],
        help="tinystories=story only; mixed=openwebtext+alpaca; conversation=tinystories+ultrachat+alpaca",
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
        help="[conversation] Relative weight for TinyStories (simple narrative).",
    )
    parser.add_argument(
        "--conv_dialog",
        type=float,
        default=0.35,
        help="[conversation] Relative weight for UltraChat (multi-turn chat).",
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
        help="[conversation] HuggingFace split for UltraChat (train_sft, train_gen, test_sft, test_gen).",
    )
    parser.add_argument(
        "--early-exit-after",
        type=int,
        default=None,
        help="Run auxiliary LM head after this many TransformerBlocks (must be < n_layers). Default from BrainConfig.",
    )
    parser.add_argument(
        "--early-exit-loss-weight",
        type=float,
        default=None,
        help="Auxiliary CE on early logits; 0 disables. Default from BrainConfig (e.g. 0.25).",
    )
    args = parser.parse_args()
    train(args)
    # End of file
