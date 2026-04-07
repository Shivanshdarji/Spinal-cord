"""
SpinalCord LLM — Train a *custom* tokenizer (NOT LLaMA)
======================================================
Trains a BPE tokenizer on TinyStories (streaming) and exports it to:
  PROJECT_ROOT/hf_export/

This makes the whole stack "yours":
- your tokenizer
- your Brain weights
- your Draft (SpinalCord) weights

Run:
  python train/train_tokenizer.py --vocab_size 32000 --max_samples 200000
"""

from __future__ import annotations

import os
import argparse
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


def train_bpe_tokenizer(lines: Iterable[str], vocab_size: int, out_dir: Path) -> None:
    try:
        from tokenizers import Tokenizer  # type: ignore
        from tokenizers.models import BPE  # type: ignore
        from tokenizers.trainers import BpeTrainer  # type: ignore
        from tokenizers.pre_tokenizers import ByteLevel  # type: ignore
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder  # type: ignore
        from tokenizers.processors import TemplateProcessing  # type: ignore
    except Exception as e:
        raise RuntimeError(f"tokenizers not installed. Run: pip install tokenizers\nError: {e}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Special tokens: keep them explicit for CausalLM
    special = ["<unk>", "<s>", "</s>", "<pad>"]

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=int(vocab_size),
        # Use 1 to reliably reach vocab_size on small-ish corpora.
        min_frequency=1,
        special_tokens=special,
        # Ensure a complete byte-level alphabet so the tokenizer can always
        # reach the requested vocab size deterministically.
        initial_alphabet=ByteLevel.alphabet(),
    )

    tok.train_from_iterator(lines, trainer=trainer)

    # Add BOS/EOS post-processing so models can optionally use them later.
    # We still train with add_special_tokens=False in encode().
    bos_id = tok.token_to_id("<s>")
    eos_id = tok.token_to_id("</s>")
    tok.post_processor = TemplateProcessing(
        single="$A",
        pair="$A $B",
        special_tokens=[("<s>", bos_id), ("</s>", eos_id)],
    )

    tokenizer_json = out_dir / "tokenizer.json"
    tok.save(str(tokenizer_json))

    # Export a HF-compatible wrapper via transformers fast tokenizer
    try:
        from transformers import PreTrainedTokenizerFast  # type: ignore
    except Exception as e:
        raise RuntimeError(f"transformers not installed. Run: pip install transformers\nError: {e}")

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json),
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    hf_tok.save_pretrained(str(out_dir))

    (out_dir / "SPINALCORD_TOKENIZER.txt").write_text(
        f"tokenizer=custom_bpe\nvocab_size={len(hf_tok)}\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train custom BPE tokenizer for SpinalCord")
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--max_samples", type=int, default=200000, help="How many TinyStories samples to stream")
    ap.add_argument("--out_dir", type=str, default=str(project_root() / "hf_export"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    print("=" * 60)
    print("  SpinalCord — Train Custom Tokenizer (BPE)")
    print(f"  vocab_size:   {args.vocab_size}")
    print(f"  max_samples:  {args.max_samples:,}")
    print(f"  out_dir:      {out_dir}")
    print("=" * 60)

    lines = stream_tinystories_text(args.max_samples)
    train_bpe_tokenizer(lines, args.vocab_size, out_dir)

    print("\nTokenizer trained and exported.")
    print(f"   {out_dir}")
    print("Next:")
    print("  python train/train_brain.py --steps 6000")
    print("  python train/distill_draft.py --brain_ckpt models/scbrain_best.pt --steps 5000")
    print("  python convert/convert_both.py")


if __name__ == "__main__":
    main()

