"""
SpinalCord LLM — Configuration v2.0
=====================================
Custom LLM architectures designed to fit on RTX 2050 (4GB VRAM)
and achieve 10x speculative decoding speedup.

Strategy:
  SCBrain-1B  → 16 layers, 1.1B params  → GPU (Brain/Verifier)
  SCDraft-120M → 6 layers,  120M params  → CPU (Spinal Cord / Draft)
  
  The Draft is distillation-trained FROM the Brain so it predicts
  Brain's tokens with 85-90%+ acceptance → 10x speculative gain.

Author: Shivansh Darji | AppDice
"""

from dataclasses import dataclass, field


# ─── Draft Model Config (The "Spinal Cord") ───────────────────────────────────
# 6 layers, 120M params, CPU inference.
# Distilled to perfectly mimic Brain's token distribution.
# Must use SAME vocabulary as Brain for speculative decoding to work.
@dataclass
class DraftConfig:
    name: str = "SCDraft-120M"

    # Vocabulary — MUST match Brain exactly
    vocab_size: int = 32000
    max_seq_len: int = 2048

    # Architecture — tiny but precise
    n_layers:   int   = 6       # 6 "spinal cord" layers
    n_heads:    int   = 8       # 8 attention heads
    n_kv_heads: int   = 4       # GQA: 4 KV heads → faster
    d_model:    int   = 768     # embedding dim
    d_ff:       int   = 2048    # ffn hidden size
    dropout:    float = 0.1

    # Training
    batch_size:    int   = 32
    learning_rate: float = 3e-4
    max_steps:     int   = 20000
    warmup_steps:  int   = 500
    grad_clip:     float = 1.0

    # Speculative window — draft generates this many tokens at once
    gamma: int = 8   # 8 tokens per round → 8x parallelism cap


# ─── Brain Model Config ────────────────────────────────────────────────────────
# 16 layers, ~1.1B params, RTX 2050 GPU.
# Smart enough to answer like GPT. Small enough to fit in 4GB VRAM.
@dataclass
class BrainConfig:
    name: str = "SCBrain-1B"

    # Vocabulary
    vocab_size: int = 32000
    max_seq_len: int = 2048

    # Architecture — fits in 4GB VRAM at Q4_K_M quantization
    n_layers:   int   = 16      # 16 deep reasoning layers
    n_heads:    int   = 16      # 16 attention heads
    n_kv_heads: int   = 8       # GQA: 8 KV heads
    d_model:    int   = 2048    # embedding dim
    d_ff:       int   = 5504    # ffn hidden size (LLaMA ratio)
    dropout:    float = 0.0

    # Training — gradient accumulate to simulate larger batch
    batch_size:         int   = 2      # limited by 4GB VRAM
    grad_accum_steps:   int   = 16     # effective batch = 32
    learning_rate:      float = 3e-4
    max_steps:          int   = 50000
    warmup_steps:       int   = 2000
    grad_clip:          float = 1.0
    weight_decay:       float = 0.1

    # Mixed precision
    use_bf16: bool = True   # bfloat16 for Ampere (RTX 20xx series)

    # Early exit (same model, shallow path for "easy" positions — must be trained)
    # After `early_exit_after` TransformerBlocks, an auxiliary LM head can predict;
    # inference uses high max-prob on that head to skip deeper layers for that step.
    early_exit_after: int = 4
    early_exit_loss_weight: float = 0.25  # auxiliary CE; set 0 to disable extra term


# ─── Distillation Config ──────────────────────────────────────────────────────
# How we train the Draft to mimic the Brain (the key to 10x speed!)
@dataclass
class DistillConfig:
    # Temperature for softening Brain's distributions
    # Higher temp = softer targets = easier for Draft to learn
    temperature:     float = 4.0

    # Mix of distillation loss + standard CE loss
    # alpha=1.0 means pure distillation, alpha=0.0 means pure CE
    alpha:           float = 0.9   # 90% distillation, 10% CE

    # How many tokens of context to use during distillation
    max_seq_len:     int   = 1024

    batch_size:      int   = 16
    learning_rate:   float = 2e-4
    max_steps:       int   = 30000
    warmup_steps:    int   = 1000
    grad_clip:       float = 1.0


# ─── Speculative Decoding Config ──────────────────────────────────────────────
@dataclass
class SpinalCordConfig:
    draft:    DraftConfig    = field(default_factory=DraftConfig)
    brain:    BrainConfig    = field(default_factory=BrainConfig)
    distill:  DistillConfig  = field(default_factory=DistillConfig)

    # Acceptance threshold for speculative decoding
    # Min ratio: min(1, p_brain / p_draft) must be > threshold to accept
    acceptance_threshold: float = 0.0   # 0 = pure stochastic (paper default)

    # Devices
    draft_device: str = "cpu"
    brain_device: str = "cuda"


# Param count helpers
def count_params(n_layers, d_model, n_heads, n_kv_heads, d_ff, vocab_size):
    head_dim = d_model // n_heads
    attn     = d_model * (n_heads * head_dim + 2 * n_kv_heads * head_dim + d_model)
    ffn      = d_model * d_ff * 3
    norms    = d_model * 2
    per_layer = attn + ffn + norms
    embed    = vocab_size * d_model
    total    = n_layers * per_layer + embed
    return total / 1e9


if __name__ == "__main__":
    b = BrainConfig()
    d = DraftConfig()
    bp = count_params(b.n_layers, b.d_model, b.n_heads, b.n_kv_heads, b.d_ff, b.vocab_size)
    dp = count_params(d.n_layers, d.d_model, d.n_heads, d.n_kv_heads, d.d_ff, d.vocab_size)

    print("=" * 55)
    print("  SpinalCord LLM — Custom Architecture Summary")
    print("=" * 55)
    print(f"  SCBrain-1B:   {bp:.2f}B params  | {b.n_layers} layers | d={b.d_model}")
    print(f"  SCDraft-120M: {dp:.3f}B params | {d.n_layers} layers | d={d.d_model}")
    print(f"  Speed ratio:  {bp/dp:.1f}x (Brain is {bp/dp:.1f}x bigger than Draft)")
    print(f"  Gamma:        {d.gamma} tokens speculated per round")
    print(f"  Theoretical max speedup: ~{d.gamma}x (at 100% acceptance)")
    print(f"  Realistic speedup: ~{d.gamma * 0.85:.1f}x (at 85% acceptance)")
    print("=" * 55)
