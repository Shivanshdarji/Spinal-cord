"""
SpinalCord LLM — Training Loop
=================================
Trains the Draft (Spinal Cord) model on text data.
Uses AdamW optimizer with cosine LR schedule.

Run:
    python train.py

Author: Shivansh Darji | AppDice
"""

import time
import math
import json
import torch
import torch.nn.functional as F
from pathlib import Path

from config import SpinalCordConfig
from model import SpinalCordDraft
from dataset import load_tinystories, get_dataloaders


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
cfg = SpinalCordConfig()
draft_cfg = cfg.draft

# Output dir
OUT_DIR = Path("../checkpoints")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT_DIR / "training_log.json"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Train] Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Train] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ─────────────────────────────────────────────────────────────────────────────
# COSINE LR SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr_ratio: float = 0.1) -> float:
    min_lr = max_lr * min_lr_ratio
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    if step >= max_steps:
        return min_lr
    denom = max(max_steps - warmup_steps, 1)
    progress = (step - warmup_steps) / denom
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING STEP
# ─────────────────────────────────────────────────────────────────────────────
# Removed @torch.compile (Triton is not natively supported on Windows)
def train_step(model, x, y):
    logits = model(x)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
    )
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, val_loader, max_batches: int = 50) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    for x, y in val_loader:
        if count >= max_batches:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  🧠 SpinalCord LLM — Training Draft Model")
    print("  AppDice | Shivansh Darji")
    print("=" * 60)

    # ── Load Data ────────────────────────────────────────────────────────────
    all_tokens = load_tinystories(max_tokens=5_000_000)
    split = int(0.9 * len(all_tokens))
    train_tokens = all_tokens[:split]
    val_tokens   = all_tokens[split:]

    train_loader, val_loader = get_dataloaders(
        train_tokens, val_tokens,
        seq_len=draft_cfg.max_seq_len,
        batch_size=draft_cfg.batch_size,
    )

    # ── Build Model ──────────────────────────────────────────────────────────
    model = SpinalCordDraft(draft_cfg).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n[Train] Model parameters: {total_params:.1f}M")
    print(f"[Train] Architecture:")
    print(f"         Layers:    {draft_cfg.n_layers}")
    print(f"         d_model:   {draft_cfg.d_model}")
    print(f"         n_heads:   {draft_cfg.n_heads}")
    print(f"         n_kv_heads:{draft_cfg.n_kv_heads}")
    print(f"         d_ff:      {draft_cfg.d_ff}\n")

    # ── Optimizer ────────────────────────────────────────────────────────────
    # Separate weight decay: don't decay biases and norms
    decay_params  = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    
    optimizer = torch.optim.AdamW([
        {"params": decay_params,   "weight_decay": 0.1},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=draft_cfg.learning_rate, betas=(0.9, 0.95))

    # ── AMP Scaler (Mixed Precision) ─────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    # ── Training Log ─────────────────────────────────────────────────────────
    log = {"steps": [], "train_loss": [], "val_loss": [], "lr": []}

    # ── Loop ─────────────────────────────────────────────────────────────────
    model.train()
    step = 0
    train_iter = iter(train_loader)
    best_val_loss = float("inf")
    t0 = time.time()

    print("[Train] Starting training loop...\n")
    
    while step < draft_cfg.max_steps:
        # Get batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        
        x, y = x.to(DEVICE), y.to(DEVICE)

        # Learning rate update
        lr = get_lr(step, draft_cfg.warmup_steps, draft_cfg.max_steps, draft_cfg.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward + Backward
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            loss = train_step(model, x, y)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), draft_cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        step += 1

        # ── Logging ──────────────────────────────────────────────────────────
        if step % 100 == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            
            # Tokens per second
            tokens_per_sec = 100 * draft_cfg.batch_size * draft_cfg.max_seq_len / dt
            
            print(f"Step {step:5d}/{draft_cfg.max_steps} | "
                  f"Loss: {loss.item():.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Tok/s: {tokens_per_sec:,.0f}")

        # ── Validation ───────────────────────────────────────────────────────
        if step % 500 == 0:
            val_loss = evaluate(model, val_loader)
            print(f"\n{'─'*50}")
            print(f"  ✅ Validation | Step {step} | Loss: {val_loss:.4f}")
            
            log["steps"].append(step)
            log["train_loss"].append(loss.item())
            log["val_loss"].append(val_loss)
            log["lr"].append(lr)
            
            with open(LOG_FILE, "w") as f:
                json.dump(log, f, indent=2)
            
            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = OUT_DIR / "spinalcord_draft_best.pt"
                torch.save({
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": draft_cfg,
                }, ckpt_path)
                print(f"  💾 Saved best checkpoint: {ckpt_path}")
            print(f"{'─'*50}\n")

    print("\n✨ Training complete!")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Checkpoint: {OUT_DIR / 'spinalcord_draft_best.pt'}")


if __name__ == "__main__":
    main()
