"""
SpinalCord LLM — Train a custom SentencePiece tokenizer (llama.cpp compatible)
=============================================================================
llama.cpp's HF→GGUF conversion reliably supports SentencePiece `tokenizer.model`.

This script streams TinyStories from HuggingFace, writes a temporary training text,
trains SentencePiece, and exports:
  PROJECT_ROOT/hf_export/tokenizer.model

Run:
  python train/train_tokenizer_spm.py --vocab_size 32000 --max_samples 200000
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Iterable


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def stream_tinystories_text(max_samples: int) -> Iterable[str]:
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except Exception as e:
        raise RuntimeError(f"datasets not installed. Run: pip install datasets\nError: {e}")

    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    n = 0
    for sample in ds:
        text = str(sample.get("text", "")).strip()
        if text:
            yield text
            n += 1
            if n >= max_samples:
                break


def train_sentencepiece(lines: Iterable[str], vocab_size: int, out_dir: Path) -> Path:
    try:
        import sentencepiece as spm  # type: ignore
    except Exception as e:
        raise RuntimeError(f"sentencepiece not installed. Run: pip install sentencepiece\nError: {e}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write a temporary corpus file
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".txt") as f:
        corpus_path = f.name
        for line in lines:
            f.write(line.replace("\n", " ").strip() + "\n")

    prefix = str(out_dir / "tokenizer")
    # Train SentencePiece BPE; reserve special ids compatible with llama.cpp conventions.
    # <unk>=0, <s>=1, </s>=2, <pad>=3
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=prefix,
        vocab_size=int(vocab_size),
        model_type="bpe",
        character_coverage=1.0,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=3,
        user_defined_symbols=[],
        byte_fallback=True,
        normalization_rule_name="identity",
    )

    # Cleanup temp corpus
    try:
        os.remove(corpus_path)
    except Exception:
        pass

    model_path = out_dir / "tokenizer.model"
    (out_dir / "SPINALCORD_TOKENIZER.txt").write_text(
        f"tokenizer=sentencepiece_bpe\nvocab_size={vocab_size}\n",
        encoding="utf-8",
    )
    return model_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Train SentencePiece tokenizer for SpinalCord")
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--max_samples", type=int, default=200000)
    ap.add_argument("--out_dir", type=str, default=str(project_root() / "hf_export"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    print("=" * 60)
    print("  SpinalCord — Train Custom Tokenizer (SentencePiece)")
    print(f"  vocab_size:   {args.vocab_size}")
    print(f"  max_samples:  {args.max_samples:,}")
    print(f"  out_dir:      {out_dir}")
    print("=" * 60)

    model_path = train_sentencepiece(stream_tinystories_text(args.max_samples), args.vocab_size, out_dir)
    print("\nTokenizer trained and exported.")
    print(f"   {model_path}")
    print("Next (REQUIRES retraining models with this tokenizer):")
    print("  python train/train_brain.py --steps 6000")
    print("  python train/distill_draft.py --brain_ckpt models/scbrain_best.pt --steps 5000")
    print("  python convert/convert_both.py")


if __name__ == "__main__":
    main()

