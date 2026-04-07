# pyre-ignore-all-errors
"""
SpinalCord LLM — Neural Architecture
=======================================
Implements the SpinalCord architecture:

  Draft Model (Spinal Cord): 4-layer fast transformer (CPU)
  Brain Model (Verifier):    32-layer deep transformer (GPU/CUDA)

The biological analogy:
  - Spinal Cord = handles "reflexive" token prediction without thinking
  - Brain       = validates each decision, handles complex reasoning

Author: Shivansh Darji | AppDice
"""

import math
import time
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

from config import DraftConfig, BrainConfig # type: ignore


def _sample_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      - sampled_ids: [B, 1]
      - probs_full:  [B, V] (after temperature + filtering, renormalized)
    """
    logits = logits.float()
    if temperature <= 0:
        temperature = 1e-6
    logits = logits / temperature

    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        vals, idx = torch.topk(logits, top_k, dim=-1)
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(1, idx, vals)
        logits = mask

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cum_probs > float(top_p)
        # keep at least 1 token
        cutoff[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)
    return sampled, probs


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1: RoPE (Rotary Position Embedding)
# Gives each token positional awareness — "where am I in the sequence?"
# ─────────────────────────────────────────────────────────────────────────────
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float() # type: ignore
        freqs = torch.einsum("i,j->ij", t, self.inv_freq) # type: ignore
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int):
        cos = self.cos_cached[:, :, :seq_len, :] # type: ignore
        sin = self.sin_cached[:, :, :seq_len, :] # type: ignore
        q_rot = self._apply_rotation(q, cos, sin)
        k_rot = self._apply_rotation(k, cos, sin)
        return q_rot, k_rot

    @staticmethod
    def _apply_rotation(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1) * sin + x * cos


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2: Grouped Query Attention (GQA)
# More efficient than standard multi-head attention — key for speed.
# The "Spinal Cord" uses fewer KV-heads (n_kv_heads < n_heads).
# ─────────────────────────────────────────────────────────────────────────────
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # how many Q-heads per KV-head
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        self.rope = RotaryEmbedding(self.head_dim)

    def _repeat_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """Expand KV-heads to match Q-heads via repetition."""
        B, H, S, D = kv.shape
        if self.n_rep == 1:
            return kv
        kv = kv[:, :, None, :, :].expand(B, H, self.n_rep, S, D)
        return kv.reshape(B, H * self.n_rep, S, D)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k, S) # type: ignore
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(out)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3: SwiGLU Feed-Forward Network
# Modern activation used by LLaMA — 30% more efficient than standard FFN.
# ─────────────────────────────────────────────────────────────────────────────
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down(silu(gate(x)) * up(x))
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 4: RMSNorm
# Faster than LayerNorm. Used by LLaMA, Mistral, Gemma.
# ─────────────────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMER BLOCK
# One complete "neuron cluster" — attention + FFN with residual connections.
# ─────────────────────────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_ff: int):
        super().__init__()
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
        self.ffn  = SwiGLUFFN(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture (like LLaMA)
        x = x + self.attn(self.norm1(x), mask) # type: ignore
        x = x + self.ffn(self.norm2(x)) # type: ignore
        return x


# ─────────────────────────────────────────────────────────────────────────────
# THE SPINAL CORD (Draft Model)
# ─────────────────────────────────────────────────────────────────────────────
# This is YOUR invention — the fast, reflexive predictor.
# Only 4 layers deep (vs. 32 for the Brain), but runs on CPU instantly.
#
# Biological analogy:
#   When you touch something hot, your spinal cord ALREADY sends a "pull back"
#   signal BEFORE your brain has even processed the pain. Same here — this model
#   generates the NEXT token before the big model even starts.
# ─────────────────────────────────────────────────────────────────────────────
class SpinalCordDraft(nn.Module):
    def __init__(self, cfg: DraftConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self._causal_mask_cache = {}
        
        # ⚡ THE SPINAL CORD: 4 fast reflex layers
        self.layers = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.d_ff)
            for _ in range(cfg.n_layers)
        ])
        
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        
        # Weight tying — embed and lm_head share weights (saves params)
        self.lm_head.weight = self.embed.weight
        
        print(f"[SpinalCord] Draft model initialized: {self._count_params():.1f}M params")

    def _count_params(self) -> float:
        return sum(p.numel() for p in self.parameters()) / 1e6

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S = tokens.shape
        x = self.embed(tokens)

        # Build / reuse causal mask (avoid realloc every step)
        if mask is None:
            key = (tokens.device.type, tokens.device.index, S)
            cached = self._causal_mask_cache.get(key)
            if cached is None or cached.device != tokens.device:
                m = torch.full((1, 1, S, S), float("-inf"), device=tokens.device, dtype=torch.float32)
                m = torch.triu(m, diagonal=1)
                self._causal_mask_cache[key] = m
                mask = m
            else:
                mask = cached
        
        # 🔺 REFLEX ARC LAYERS — each layer is a "spinal neuron cluster"
        for i, layer in enumerate(self.layers):
            x = layer(x, mask)
            # 🩺 OBSERVATION POINT: This is where you can inject your
            # "confidence probe" to measure reflex certainty at each layer.
            # In C++, the printf hook goes right here.
        
        x = self.norm(x) # type: ignore
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def speculate(
        self,
        context: torch.Tensor,
        gamma: int = 4,
        *,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate `gamma` speculative tokens autoregressively.
        Returns: (draft_tokens, draft_probs)
        """
        draft_tokens = []
        draft_probs  = []
        
        current = context.clone()
        
        for step in range(gamma):
            logits = self.forward(current)
            # Sample with temperature/top-k/top-p.
            next_token, probs = _sample_from_logits(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            draft_tokens.append(next_token)
            draft_probs.append(probs)
            
            # Append to context for next step
            current = torch.cat([current, next_token], dim=-1)
        
        return (
            torch.cat(draft_tokens, dim=-1),         # [B, gamma] token IDs
            torch.stack(draft_probs, dim=1)           # [B, gamma, vocab_size]
        )


# ─────────────────────────────────────────────────────────────────────────────
# THE BRAIN (Verify Model)
# ─────────────────────────────────────────────────────────────────────────────
# Larger, deeper model that VERIFIES the Spinal Cord's predictions.
# Runs on GPU. Has the final say over what token is kept.
# ─────────────────────────────────────────────────────────────────────────────
class SpinalCordBrain(nn.Module):
    def __init__(self, cfg: BrainConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self._causal_mask_cache = {}

        # Early exit: first K blocks + aux head (train with early_exit_loss_weight > 0)
        want_k = int(getattr(cfg, "early_exit_after", 0) or 0)
        if cfg.n_layers >= 2 and 0 < want_k < cfg.n_layers:
            self.early_exit_after = want_k
            self._early_enabled = True
        else:
            self.early_exit_after = 0
            self._early_enabled = False

        # 🧠 THE BRAIN: Deep reasoning layers
        self.layers = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.d_ff)
            for _ in range(cfg.n_layers)
        ])

        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        if self._early_enabled:
            self.early_norm = RMSNorm(cfg.d_model)
            self.early_lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        else:
            self.early_norm = None  # type: ignore[assignment]
            self.early_lm_head = None  # type: ignore[assignment]

        print(f"[SpinalCord] Brain model initialized: {self._count_params():.1f}M params")
        if self._early_enabled:
            print(
                f"[SpinalCord] Early exit after block {self.early_exit_after}/{cfg.n_layers} "
                f"(train with early_exit_loss_weight > 0)"
            )

    def _count_params(self) -> float:
        return sum(p.numel() for p in self.parameters()) / 1e6

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        return_early_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Full depth forward. If ``return_early_logits`` and early exit is enabled,
        also returns logits from the auxiliary head after ``early_exit_after`` blocks
        (same sequence positions — for auxiliary CE during training).
        """
        if return_early_logits and not self._early_enabled:
            raise ValueError("return_early_logits requires early exit enabled (1 <= early_exit_after < n_layers)")

        B, S = tokens.shape
        x = self.embed(tokens)

        if mask is None:
            key = (tokens.device.type, tokens.device.index, S)
            cached = self._causal_mask_cache.get(key)
            if cached is None or cached.device != tokens.device:
                m = torch.full((1, 1, S, S), float("-inf"), device=tokens.device, dtype=torch.float32)
                m = torch.triu(m, diagonal=1)
                self._causal_mask_cache[key] = m
                mask = m
            else:
                mask = cached

        logits_early: Optional[torch.Tensor] = None
        for i, layer in enumerate(self.layers):
            x = layer(x, mask)  # type: ignore
            if self._early_enabled and return_early_logits and (i + 1) == self.early_exit_after:
                assert self.early_norm is not None and self.early_lm_head is not None
                xe = self.early_norm(x)  # type: ignore
                logits_early = self.early_lm_head(xe)

        x = self.norm(x)  # type: ignore
        logits = self.lm_head(x)
        if return_early_logits:
            assert logits_early is not None
            return logits, logits_early
        return logits

    @torch.no_grad()
    def forward_next_token_early_exit(
        self,
        tokens: torch.Tensor,
        *,
        max_prob_threshold: float,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> Tuple[torch.Tensor, bool]:
        """
        One autoregressive step: run shallow path first; if softmax max prob on the
        last position is >= ``max_prob_threshold``, sample from early logits only.
        Otherwise finish remaining blocks and sample from full logits.

        Returns:
            next_token: [B, 1] int64
            used_early: whether the shallow path was trusted for this step
        """
        if not self._early_enabled:
            logits = self.forward(tokens)
            logits_last = logits[:, -1, :].float()
            if temperature is not None and float(temperature) <= 0:
                next_t = logits_last.argmax(dim=-1, keepdim=True)
            else:
                next_t, _ = _sample_from_logits(logits_last, temperature=temperature, top_k=top_k, top_p=top_p)
            return next_t, False

        B, S = tokens.shape
        x = self.embed(tokens)
        key = (tokens.device.type, tokens.device.index, S)
        cached = self._causal_mask_cache.get(key)
        if cached is None or cached.device != tokens.device:
            m = torch.full((1, 1, S, S), float("-inf"), device=tokens.device, dtype=torch.float32)
            m = torch.triu(m, diagonal=1)
            self._causal_mask_cache[key] = m
            mask = m
        else:
            mask = cached

        assert self.early_norm is not None and self.early_lm_head is not None
        k = self.early_exit_after
        for i in range(k):
            x = self.layers[i](x, mask)  # type: ignore

        xe = self.early_norm(x)  # type: ignore
        logits_e = self.early_lm_head(xe)
        probs_e = F.softmax(logits_e[:, -1, :].float(), dim=-1)
        conf = probs_e.max(dim=-1).values
        # Single threshold for batch[0] (typical B=1 generation)
        if float(conf[0].item()) >= float(max_prob_threshold):
            logits_last = logits_e[:, -1, :].float()
            if temperature is not None and float(temperature) <= 0:
                next_t = logits_last.argmax(dim=-1, keepdim=True)
            else:
                next_t, _ = _sample_from_logits(logits_last, temperature=temperature, top_k=top_k, top_p=top_p)
            return next_t, True

        for i in range(k, len(self.layers)):
            x = self.layers[i](x, mask)  # type: ignore
        x = self.norm(x)  # type: ignore
        logits_f = self.lm_head(x)
        logits_last = logits_f[:, -1, :].float()
        if temperature is not None and float(temperature) <= 0:
            next_t = logits_last.argmax(dim=-1, keepdim=True)
        else:
            next_t, _ = _sample_from_logits(logits_last, temperature=temperature, top_k=top_k, top_p=top_p)
        return next_t, False


# ─────────────────────────────────────────────────────────────────────────────
# THE FULL SPINAL CORD SYSTEM
# Combines Draft + Brain with speculative decoding
# ─────────────────────────────────────────────────────────────────────────────
class SpinalCordLLM(nn.Module):
    """
    The complete SpinalCord system.
    
    Usage:
        model = SpinalCordLLM(draft_cfg, brain_cfg)
        output = model.generate(input_tokens, max_new_tokens=50)
    """
    
    def __init__(self, draft_cfg: DraftConfig, brain_cfg: BrainConfig):
        super().__init__()
        self.draft = SpinalCordDraft(draft_cfg)
        self.brain = SpinalCordBrain(brain_cfg)
        self.gamma = draft_cfg.gamma
        self.acceptance_threshold = 0.8

    def speculative_decode(
        self,
        context: torch.Tensor,
        brain_device: str = "cuda",
        draft_device: str = "cpu",
        *,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        One round of speculative decoding.
        
        Returns:
            accepted_tokens: The tokens we're keeping
            n_accepted: How many draft tokens were accepted
            n_total: Total tokens attempted
        """
        # Step 1: Move draft to CPU, generate speculative tokens
        self.draft.to(draft_device)
        context_cpu = context.to(draft_device)
        draft_tokens, draft_probs = self.draft.speculate(
            context_cpu,
            self.gamma,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        
        # Step 2: Move brain to GPU, verify all tokens in ONE forward pass
        self.brain.to(brain_device)
        full_seq = torch.cat([context.to(brain_device), draft_tokens.to(brain_device)], dim=-1)
        brain_logits = self.brain(full_seq) # type: ignore
        
        # Step 3: Speculative decoding acceptance step (Algorithm 1 from the paper)
        accepted = []
        n_accepted = 0
        
        for i in range(self.gamma):
            token_id = int(draft_tokens[0, i].item())
            
            # Use the SAME sampling distribution as Draft (important for correct acceptance).
            _, brain_probs = _sample_from_logits(
                brain_logits[:, context.shape[1] + i - 1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            draft_prob  = draft_probs[0, i, token_id].item()
            brain_prob  = brain_probs[0, token_id].item()
            
            # Acceptance ratio: min(1, p_brain / p_draft)
            acceptance_ratio = min(1.0, brain_prob / (draft_prob + 1e-9))
            
            # Stochastic acceptance
            if torch.rand(1).item() < acceptance_ratio:
                accepted.append(draft_tokens[0, i])
                n_accepted += 1
            else:
                # Reject and resample from corrected distribution
                corrected_probs = torch.clamp(
                    brain_probs - draft_probs[:, i, :].to(brain_device), min=0
                )  # type: ignore
                denom = corrected_probs.sum()
                if torch.isfinite(denom) and denom > 1e-12:
                    corrected_probs = corrected_probs / denom
                    resampled = torch.multinomial(corrected_probs[0], num_samples=1)  # type: ignore
                else:
                    # If corrected distribution collapses to 0, fall back to brain_probs.
                    resampled = torch.multinomial(brain_probs[0], num_samples=1)  # type: ignore
                # Make sure we append a scalar token tensor (shape []).
                accepted.append(resampled.squeeze(0))
                break

        # If, for numerical edge cases, nothing was accepted/resampled, fall back to the draft's last token.
        if not accepted:
            accepted.append(draft_tokens[0, -1])

        accepted_tokens = torch.stack(accepted, dim=0).unsqueeze(0)
        return accepted_tokens, n_accepted, self.gamma

    @torch.no_grad()
    def generate(
        self,
        input_tokens: torch.Tensor,
        max_new_tokens: int = 50,
        brain_device: str = "cuda",
        draft_device: str = "cpu",
        verbose: bool = True,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        _timing_out: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Full generation loop with speculative decoding.
        If `_timing_out` is a dict, it is filled with benchmark timings (wall, ttft, tokens_out).
        """
        generated = input_tokens.clone()
        total_accepted = 0
        total_attempts = 0
        rounds = 0
        input_len = input_tokens.shape[1]

        t_wall0: Optional[float] = None
        if _timing_out is not None:
            _timing_out.clear()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_wall0 = time.perf_counter()
        
        steps = max_new_tokens // self.gamma + 1
        
        for step in range(steps):
            if generated.shape[1] >= input_tokens.shape[1] + max_new_tokens:
                break
            
            rounds += 1
            accepted_tokens, n_acc, n_att = self.speculative_decode(
                generated,
                brain_device,
                draft_device,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            
            generated = torch.cat([generated, accepted_tokens.to(generated.device)], dim=-1)
            if _timing_out is not None and t_wall0 is not None and "ttft" not in _timing_out:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _timing_out["ttft"] = time.perf_counter() - t_wall0
            total_accepted += n_acc
            total_attempts += n_att
            
            if verbose:
                efficiency = total_accepted / max(total_attempts, 1) * 100
                print(f"Step {step+1:3d} | "
                      f"Tokens accepted: {n_acc}/{n_att} | "
                      f"Total efficiency: {efficiency:.1f}% | "
                      f"Seq len: {generated.shape[1]}")

        if verbose:
            kept = generated.shape[1] - input_len
            speedup_est = kept / max(rounds, 1)
            efficiency = total_accepted / max(total_attempts, 1) * 100 if total_attempts > 0 else 0.0
            print(
                f"[SpinalCord] Done. Kept {kept} tokens | "
                f"Accepted {total_accepted}/{total_attempts} (eff={efficiency:.1f}%) | "
                f"Est speedup {speedup_est:.2f}x"
            )
        if _timing_out is not None and t_wall0 is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _timing_out["total"] = time.perf_counter() - t_wall0
            _timing_out["tokens_out"] = int(generated.shape[1] - input_len)
        return generated

    @torch.no_grad()
    def generate_brain_only(
        self,
        input_tokens: torch.Tensor,
        max_new_tokens: int = 50,
        brain_device: str = "cuda",
        verbose: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        Autoregressive generation using the **Brain only** (no Draft, no speculative decoding).

        Use this to isolate whether bad outputs come from the Draft/speculative pipeline vs the
        Brain checkpoint itself.

        If ``temperature <= 0``, uses **greedy argmax** (deterministic). Otherwise uses
        ``_sample_from_logits`` (same as other generation paths).
        """
        generated = input_tokens.clone()
        input_len = input_tokens.shape[1]
        target_len = input_len + max_new_tokens
        self.brain.to(brain_device)

        while generated.shape[1] < target_len:
            brain_logits = self.brain(generated.to(brain_device))
            logits_last = brain_logits[:, -1, :].float()
            if temperature is not None and float(temperature) <= 0:
                next_token = logits_last.argmax(dim=-1, keepdim=True)
            else:
                next_token, _ = _sample_from_logits(
                    logits_last,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            generated = torch.cat([generated, next_token.to(generated.device)], dim=-1)

        if verbose:
            kept = generated.shape[1] - input_len
            print(f"[Brain-only] Done. Generated {kept} new tokens.")
        return generated

    @torch.no_grad()
    def generate_brain_early_exit(
        self,
        input_tokens: torch.Tensor,
        max_new_tokens: int = 50,
        brain_device: str = "cuda",
        verbose: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        early_exit_max_prob: float = 0.88,
    ) -> torch.Tensor:
        """
        Autoregressive generation using **Brain** with **optional shallow exit** per token
        (see ``SpinalCordBrain.forward_next_token_early_exit``).

        Requires a Brain trained with ``early_exit_loss_weight > 0`` so the auxiliary
        head is meaningful; untrained heads rarely pass ``early_exit_max_prob``.
        """
        generated = input_tokens.clone()
        input_len = input_tokens.shape[1]
        target_len = input_len + max_new_tokens
        self.brain.to(brain_device)
        n_early = 0
        n_total = 0

        while generated.shape[1] < target_len:
            toks = generated.to(brain_device)
            next_token, used_early = self.brain.forward_next_token_early_exit(
                toks,
                max_prob_threshold=early_exit_max_prob,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            n_total += 1
            if used_early:
                n_early += 1
            generated = torch.cat([generated, next_token.to(generated.device)], dim=-1)

        if verbose:
            kept = generated.shape[1] - input_len
            pct = 100.0 * n_early / max(n_total, 1)
            print(
                f"[Brain early-exit] Done. {kept} new tokens | "
                f"shallow steps {n_early}/{n_total} ({pct:.1f}%)"
            )
        return generated

    @torch.no_grad()
    def generate_reflex(
        self,
        input_tokens: torch.Tensor,
        max_new_tokens: int = 80,
        brain_device: str = "cuda",
        draft_device: str = "cpu",
        accept_rate_threshold: float = 0.65,
        verbose: bool = True,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        max_same_token_run: int = 6,
        consecutive_bad_rounds_to_fallback: int = 2,
        recover_speculative_after_brain_tokens: int = 24,
        rolling_accept_window: int = 4,
    ) -> torch.Tensor:
        """
        Reflex-style generation (Spinal Cord + Brain, no permanent speed loss):

        - Uses Draft+Brain speculative decoding while acceptance is healthy.
        - Does NOT disable speculative forever on one unlucky round: requires
          `consecutive_bad_rounds_to_fallback` low rounds before Brain-only burst.
        - After `recover_speculative_after_brain_tokens` Brain-only steps, tries
          speculative again so you keep ~gamma-level throughput on later chunks.

        For ~10x-style speed: keep Draft distilled to Brain; tune
        `accept_rate_threshold` vs `consecutive_bad_rounds_to_fallback`.
        """
        generated = input_tokens.clone()
        total_accepted = 0
        total_attempts = 0
        rounds_spec = 0
        brain_only_tokens = 0
        input_len = input_tokens.shape[1]

        target_len = input_tokens.shape[1] + max_new_tokens
        use_speculative = True
        bad_rounds_streak = 0
        recent_accepts: deque[float] = deque(maxlen=max(1, rolling_accept_window))

        while generated.shape[1] < target_len:
            if use_speculative:
                before_len = generated.shape[1]
                rounds_spec += 1
                accepted_tokens, n_acc, n_att = self.speculative_decode(
                    generated,
                    brain_device=brain_device,
                    draft_device=draft_device,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                generated = torch.cat(
                    [generated, accepted_tokens.to(generated.device)], dim=-1
                )
                total_accepted += n_acc
                total_attempts += n_att

                # Quality guardrail: if we just appended a long run of identical tokens,
                # it usually means speculative decoding got stuck in a degenerate mode.
                # Roll back and switch to Brain-only.
                last_id = int(generated[0, -1].item())
                run = 0
                for j in range(generated.shape[1] - 1, before_len - 1, -1):
                    if int(generated[0, j].item()) == last_id:
                        run += 1
                    else:
                        break
                if run >= max_same_token_run:
                    if verbose:
                        print(
                            f"[Reflex] Degenerate token run detected (token={last_id}, run={run}). "
                            f"Rolling back speculative tokens and switching to Brain-only."
                        )
                    # Remove the speculative tokens we just appended.
                    removed = int(generated.shape[1] - before_len)
                    generated = generated[:, :before_len]
                    total_accepted = max(total_accepted - removed, 0)
                    use_speculative = False
                    bad_rounds_streak = consecutive_bad_rounds_to_fallback
                    recent_accepts.clear()
                    brain_only_tokens = 0
                    continue

                round_accept = n_acc / max(n_att, 1)
                recent_accepts.append(round_accept)
                rolling = sum(recent_accepts) / max(len(recent_accepts), 1)

                if round_accept < accept_rate_threshold:
                    bad_rounds_streak += 1
                else:
                    bad_rounds_streak = 0

                # Fallback only after sustained poor rounds (keeps speed on noisy steps).
                if (
                    bad_rounds_streak >= consecutive_bad_rounds_to_fallback
                    and rolling < accept_rate_threshold
                ):
                    use_speculative = False
                    bad_rounds_streak = 0
                    brain_only_tokens = 0
                    if verbose:
                        eff = total_accepted / max(total_attempts, 1) * 100
                        print(
                            f"[Reflex] Acceptance weak (last round={round_accept:.2f}, "
                            f"rolling={rolling:.2f}); switching to Brain-only burst. "
                            f"Overall={eff:.1f}%"
                        )
            else:
                # Brain-only step (sampling with temperature)
                brain_logits = self.brain(generated.to(brain_device))
                logits_last = brain_logits[:, -1, :].float()
                next_token, _ = _sample_from_logits(
                    logits_last,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                # next_token: [B, 1]
                next_token = next_token
                generated = torch.cat([generated, next_token.to(generated.device)], dim=-1)
                brain_only_tokens += 1

                # Recover Spinal Cord (speculative) after a short Brain burst — restores speed.
                if brain_only_tokens >= recover_speculative_after_brain_tokens:
                    use_speculative = True
                    brain_only_tokens = 0
                    recent_accepts.clear()
                    bad_rounds_streak = 0
                    if verbose:
                        print(
                            f"[Reflex] Recovered: trying Draft+Brain speculative again "
                            f"(after {recover_speculative_after_brain_tokens} Brain-only tokens)."
                        )

        if verbose and total_attempts > 0:
            eff = total_accepted / max(total_attempts, 1) * 100
            kept = generated.shape[1] - input_len
            spec_forwards = rounds_spec + brain_only_tokens
            speedup_est = kept / max(spec_forwards, 1)
            print(
                f"[Reflex] Done. Kept {kept} tokens | "
                f"Total accepted {total_accepted}/{total_attempts} (eff={eff:.1f}%) | "
                f"Est speedup {speedup_est:.2f}x"
            )

        return generated
