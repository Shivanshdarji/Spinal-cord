"""
Shared tokenizer utilities for SpinalCord.

Speculative decoding requires Brain + Draft to use the exact same tokenizer
(same vocab, same IDs).

This project trains a *custom* tokenizer and exports it to PROJECT_ROOT/hf_export.
All training scripts then load that local tokenizer (not TinyLlama / not LLaMA).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple, Optional, Any


DEFAULT_TOKENIZER_DIRNAME = "hf_export"


@dataclass(frozen=True)
class TokenizerBundle:
    name_or_path: str
    vocab_size: int
    encode: Callable[[str], list[int]]
    decode: Callable[[list[int]], str]
    hf_tokenizer: Any


def project_root_from_train_dir() -> Path:
    # train/ is inside project root
    return Path(__file__).resolve().parent.parent


def default_tokenizer_dir() -> Path:
    return project_root_from_train_dir() / DEFAULT_TOKENIZER_DIRNAME


def load_local_tokenizer(tokenizer_dir: Optional[Path] = None) -> TokenizerBundle:
    tdir = tokenizer_dir or default_tokenizer_dir()
    spm_model = tdir / "tokenizer.model"
    tok_json = tdir / "tokenizer.json"

    # Prefer SentencePiece if available (llama.cpp-friendly)
    if spm_model.exists():
        try:
            import sentencepiece as spm  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"sentencepiece not installed. Run: pip install sentencepiece\nError: {e}")

        sp = spm.SentencePieceProcessor(model_file=str(spm_model))
        vocab_size = int(sp.get_piece_size())

        def encode(text: str) -> list[int]:
            return list(sp.encode(text, out_type=int))

        def decode(ids: list[int]) -> str:
            return sp.decode(ids)

        return TokenizerBundle(
            name_or_path=str(tdir),
            vocab_size=vocab_size,
            encode=encode,
            decode=decode,
            hf_tokenizer=sp,
        )

    # Fallback: custom HF fast tokenizer.json (works for PyTorch training, but may not be GGUF-convertible)
    if not tok_json.exists():
        raise RuntimeError(
            f"Custom tokenizer not found at {tok_json} or {spm_model}. "
            f"Run: python train/train_tokenizer_spm.py"
        )

    try:
        from transformers import PreTrainedTokenizerFast  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"transformers not installed. Run: pip install transformers\nError: {e}")

    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(tok_json),
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )

    # Ensure a sane vocab_size (some tokenizers expose len(tok) more reliably)
    vocab_size = int(getattr(tok, "vocab_size", 0) or len(tok))

    def encode(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    def decode(ids: list[int]) -> str:
        return tok.decode(ids, skip_special_tokens=True)

    return TokenizerBundle(
        name_or_path=str(tdir),
        vocab_size=vocab_size,
        encode=encode,
        decode=decode,
        hf_tokenizer=tok,
    )


def load_tokenizer_and_export(
    *,
    expected_vocab_size: Optional[int] = None,
    tokenizer_dir: Optional[Path] = None,
) -> Tuple[TokenizerBundle, Path]:
    """
    Load the locally-trained tokenizer (PROJECT_ROOT/hf_export).
    """
    export_dir = tokenizer_dir or default_tokenizer_dir()
    bundle = load_local_tokenizer(export_dir)
    if expected_vocab_size is not None and bundle.vocab_size != int(expected_vocab_size):
        raise RuntimeError(
            f"Tokenizer vocab_size mismatch: tokenizer={bundle.vocab_size} expected={expected_vocab_size}. "
            f"Speculative decoding requires identical vocab for Brain and Draft."
        )
    return bundle, export_dir

