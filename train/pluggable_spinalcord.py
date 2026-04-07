# pyre-ignore-all-errors
"""
Pluggable SpinalCord engine.

Goal: keep speculative-decoding core generic, while swapping different Brain models
(and their paired Drafts) via adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

import torch  # type: ignore

from model import _sample_from_logits  # type: ignore


class BrainAdapter(Protocol):
    """Minimal interface required by SpinalCordEngine for any verifier model."""

    vocab_size: int

    def to(self, device: str):
        ...

    def forward_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return logits shaped [B, S, V]."""
        ...


class DraftAdapter(Protocol):
    """Minimal interface required by SpinalCordEngine for any draft model."""

    vocab_size: int

    def to(self, device: str):
        ...

    def speculate(
        self,
        context_tokens: torch.Tensor,
        gamma: int,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (draft_token_ids [B,gamma], draft_probs [B,gamma,V])."""
        ...


@dataclass
class SpinalCordRuntimeConfig:
    gamma: int = 8
    acceptance_floor: float = 0.0


class TorchBrainAdapter:
    """Adapter for torch nn.Module brain models with forward(tokens)->logits."""

    def __init__(self, model: torch.nn.Module, vocab_size: int):
        self.model = model
        self.vocab_size = int(vocab_size)

    def to(self, device: str):
        self.model.to(device)
        return self

    @torch.no_grad()
    def forward_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model(tokens)


class TorchDraftAdapter:
    """Adapter for draft models exposing speculate(context,gamma,...)."""

    def __init__(self, model: torch.nn.Module, vocab_size: int):
        self.model = model
        self.vocab_size = int(vocab_size)

    def to(self, device: str):
        self.model.to(device)
        return self

    @torch.no_grad()
    def speculate(
        self,
        context_tokens: torch.Tensor,
        gamma: int,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.speculate(
            context_tokens,
            gamma,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )


class SpinalCordEngine:
    """
    Generic speculative-decoding engine.

    This class is model-agnostic if you provide compatible BrainAdapter and
    DraftAdapter implementations.
    """

    def __init__(
        self,
        brain: BrainAdapter,
        draft: DraftAdapter,
        cfg: SpinalCordRuntimeConfig,
    ):
        if brain.vocab_size != draft.vocab_size:
            raise ValueError(
                f"Vocab mismatch: brain={brain.vocab_size} draft={draft.vocab_size}. "
                "Draft/Brain tokenizer must match for speculative decoding."
            )
        self.brain = brain
        self.draft = draft
        self.cfg = cfg

    @torch.no_grad()
    def speculative_round(
        self,
        context: torch.Tensor,
        *,
        brain_device: str,
        draft_device: str,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> Tuple[torch.Tensor, int, int]:
        gamma = int(self.cfg.gamma)

        self.draft.to(draft_device)
        context_draft = context.to(draft_device)
        draft_tokens, draft_probs = self.draft.speculate(
            context_draft,
            gamma,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        self.brain.to(brain_device)
        full_seq = torch.cat([context.to(brain_device), draft_tokens.to(brain_device)], dim=-1)
        brain_logits = self.brain.forward_logits(full_seq)

        accepted = []
        n_accepted = 0

        for i in range(gamma):
            token_id = int(draft_tokens[0, i].item())
            _, brain_probs = _sample_from_logits(
                brain_logits[:, context.shape[1] + i - 1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            draft_prob = draft_probs[0, i, token_id].item()
            brain_prob = brain_probs[0, token_id].item()

            ratio = min(1.0, brain_prob / (draft_prob + 1e-9))
            if ratio < self.cfg.acceptance_floor:
                ratio = 0.0

            if torch.rand(1).item() < ratio:
                accepted.append(draft_tokens[0, i])
                n_accepted += 1
            else:
                corrected_probs = torch.clamp(
                    brain_probs - draft_probs[:, i, :].to(brain_device), min=0
                )
                denom = corrected_probs.sum()
                if torch.isfinite(denom) and denom > 1e-12:
                    corrected_probs = corrected_probs / denom
                    resampled = torch.multinomial(corrected_probs[0], num_samples=1)
                else:
                    resampled = torch.multinomial(brain_probs[0], num_samples=1)
                accepted.append(resampled.squeeze(0))
                break

        if not accepted:
            accepted.append(draft_tokens[0, -1])

        out = torch.stack(accepted, dim=0).unsqueeze(0)
        return out, n_accepted, gamma

    @torch.no_grad()
    def generate(
        self,
        input_tokens: torch.Tensor,
        *,
        max_new_tokens: int,
        brain_device: str,
        draft_device: str,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        verbose: bool = False,
    ) -> torch.Tensor:
        generated = input_tokens.clone()
        total_accepted = 0
        total_attempts = 0

        steps = max_new_tokens // max(self.cfg.gamma, 1) + 1
        for step in range(steps):
            if generated.shape[1] >= input_tokens.shape[1] + max_new_tokens:
                break
            accepted_tokens, n_acc, n_att = self.speculative_round(
                generated,
                brain_device=brain_device,
                draft_device=draft_device,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            generated = torch.cat([generated, accepted_tokens.to(generated.device)], dim=-1)
            total_accepted += n_acc
            total_attempts += n_att

            if verbose:
                eff = 100.0 * total_accepted / max(total_attempts, 1)
                print(
                    f"Step {step+1:3d} | accepted {n_acc}/{n_att} | "
                    f"overall {eff:.1f}% | seq={generated.shape[1]}"
                )

        return generated
