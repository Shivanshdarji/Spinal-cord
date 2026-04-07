# pyre-ignore-all-errors
"""
SpinalCord LLM — Dataset Utilities
=====================================
Loads and tokenizes training data.
Supports: TinyStories, OpenWebText, custom text files.

Author: Shivansh Darji | AppDice
"""

import os
import random
import torch  # type: ignore
from torch.utils.data import Dataset, DataLoader, IterableDataset  # type: ignore
from typing import Optional, List, Iterator, Tuple, Callable


class TextDataset(Dataset):
    """
    Simple character/token dataset for training.
    Can load from a .txt file or HuggingFace dataset.
    """
    
    def __init__(
        self,
        tokens: torch.Tensor,
        seq_len: int = 512,
    ):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n_chunks = (len(tokens) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        end   = start + self.seq_len
        x = self.tokens[start:end]
        y = self.tokens[start + 1:end + 1]
        return x, y


def load_tinystories(
    split: str = "train",
    max_tokens: int = 10_000_000,
    tokenizer=None,
) -> torch.Tensor:
    """
    Load TinyStories dataset — perfect for testing with small compute.
    ~2GB of simple English stories. Ideal for prototyping SpinalCord.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("Run: pip install datasets")

    print(f"[Dataset] Loading TinyStories ({split})...")
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)

    texts = []
    total = 0
    for item in ds:
        text = item["text"] + "\n"
        texts.append(text)
        total += len(text)
        if total >= max_tokens:
            break

    print(f"[Dataset] Loaded {len(texts)} stories ({total/1e6:.1f}M chars)")

    if tokenizer is None:
        raise ValueError(
            "Tokenizer is required (shared vocab for Brain + Draft). "
            "Pass a HuggingFace tokenizer, e.g. from train/tokenizer_sc.py."
        )

    # HuggingFace tokenizer (required for shared vocab across Brain + Draft)
    encoded = tokenizer(
        "\n".join(texts),
        return_tensors="pt",
        truncation=False,
    )
    return encoded["input_ids"].squeeze(0)


def get_dataloaders(
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    seq_len: int = 512,
    batch_size: int = 16,
    num_workers: int = 0,
):
    """
    Create train and validation DataLoaders.
    """
    train_ds = TextDataset(train_tokens, seq_len)
    val_ds   = TextDataset(val_tokens, seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[Dataset] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    return train_loader, val_loader
# End of file


# ─────────────────────────────────────────────────────────────────────────────
# Mixed curriculum datasets (general + instruction/Q&A) for “answer anything”
# ─────────────────────────────────────────────────────────────────────────────

def _segment_tokens_with_mask(
    *,
    tokens: List[int],
    prompt_len: int,
    encode_len: int,
    seq_len: int,
    max_segments: int = 8,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Turn a single tokenized example into (x, labels) segments of length seq_len.

    labels are next-token targets from y = tokens[1:], but any target token that
    belongs to the prompt portion (index < prompt_len) is set to -1 so loss can
    use ignore_index=-1.

    Args:
      tokens: full sequence tokens = prompt_tokens + answer_tokens
      prompt_len: number of tokens belonging to prompt/question portion
      seq_len: model training context length (x and labels length)
    """
    if len(tokens) < seq_len + 1:
        return

    n_segments = 0
    # Non-overlapping segments (start step = seq_len) to keep compute bounded.
    # This is enough for mixed-curriculum training; later you can add overlap if desired.
    step = seq_len
    max_start = len(tokens) - (seq_len + 1)
    for start in range(0, max_start + 1, step):
        if n_segments >= max_segments:
            break

        seg = tokens[start : start + seq_len + 1]
        x = torch.tensor(seg[:-1], dtype=torch.long)
        y = torch.tensor(seg[1:], dtype=torch.long)

        # y[i] predicts tokens[start + i + 1]
        # Ignore target tokens that fall inside the prompt region of the example.
        for i in range(seq_len):
            target_idx_in_example = start + i + 1
            if target_idx_in_example < prompt_len:
                y[i] = -1

        yield x, y
        n_segments += 1


class TinyStoriesSegmentsDataset(IterableDataset):
    """
    Streams TinyStories (simple English stories, child-friendly prose).
    Good for plain, clear language — pairs well with dialogue + instruction mix.
    """

    def __init__(
        self,
        encode_fn: Callable[[str], List[int]],
        seq_len: int,
        split: str = "train",
        max_examples: Optional[int] = None,
        max_tokens_per_example: int = 4096,
        max_segments_per_example: int = 8,
        dataset_name: str = "roneneldan/TinyStories",
    ):
        super().__init__()
        self.encode_fn = encode_fn
        self.seq_len = seq_len
        self.split = split
        self.max_examples = max_examples
        self.max_tokens_per_example = max_tokens_per_example
        self.max_segments_per_example = max_segments_per_example
        self.dataset_name = dataset_name

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("Run: pip install datasets")

        ds = load_dataset(self.dataset_name, split=self.split, streaming=True)
        n = 0
        for item in ds:
            text = item.get("text", None)  # type: ignore[union-attr]
            if not text:
                continue

            tokens = self.encode_fn(str(text))
            if len(tokens) > self.max_tokens_per_example:
                tokens = tokens[: self.max_tokens_per_example]

            for x, y in _segment_tokens_with_mask(
                tokens=tokens,
                prompt_len=0,
                encode_len=len(tokens),
                seq_len=self.seq_len,
                max_segments=self.max_segments_per_example,
            ):
                yield x, y

            n += 1
            if self.max_examples is not None and n >= self.max_examples:
                break


def _conversation_row_to_text(item: dict) -> str:
    """
    Turn a HF row into plain text for LM training.
    Supports common parquet-based chat sets (UltraChat `text`, OpenAI-style `messages`,
    legacy `daily_dialog` `dialog` lists if present).
    """
    if not isinstance(item, dict):
        return ""
    t = item.get("text")
    if t is not None and str(t).strip():
        return str(t).strip() + "\n"
    msgs = item.get("messages")
    if msgs and isinstance(msgs, list):
        chunks: List[str] = []
        for m in msgs:
            if isinstance(m, dict):
                role = str(m.get("role", "user")).strip()
                content = str(m.get("content", "")).strip()
                if content:
                    chunks.append(f"{role}: {content}")
            elif isinstance(m, str) and m.strip():
                chunks.append(m.strip())
        if chunks:
            return "\n".join(chunks) + "\n"
    # Legacy DailyDialog-style `dialog` (only if you load an old export)
    d = item.get("dialog")
    if not d:
        return ""
    if isinstance(d[0], str):  # type: ignore[index]
        return "\n".join(str(u).strip() for u in d if str(u).strip()) + "\n"
    lines: List[str] = []
    for turn in d:
        if isinstance(turn, list):
            for u in turn:
                s = str(u).strip()
                if s:
                    lines.append(s)
        else:
            s = str(turn).strip()
            if s:
                lines.append(s)
    return "\n".join(lines) + "\n"


class MultiTurnChatSegmentsDataset(IterableDataset):
    """
    Multi-turn conversational English (default: **UltraChat** parquet on the Hub).

    Note: The old Hub dataset id `daily_dialog` used a Python *script* loader; recent
    `datasets` versions reject script-based datasets. UltraChat streams reliably.

    UltraChat uses split names ``train_sft`` / ``train_gen`` (not ``train``).
    """

    def __init__(
        self,
        encode_fn: Callable[[str], List[int]],
        seq_len: int,
        split: str = "train_sft",
        max_examples: Optional[int] = None,
        max_tokens_per_example: int = 4096,
        max_segments_per_example: int = 8,
        dataset_name: str = "HuggingFaceH4/ultrachat_200k",
    ):
        super().__init__()
        self.encode_fn = encode_fn
        self.seq_len = seq_len
        self.split = split
        self.max_examples = max_examples
        self.max_tokens_per_example = max_tokens_per_example
        self.max_segments_per_example = max_segments_per_example
        self.dataset_name = dataset_name

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("Run: pip install datasets")

        ds = load_dataset(self.dataset_name, split=self.split, streaming=True)

        n = 0
        for item in ds:
            text = _conversation_row_to_text(item)  # type: ignore[arg-type]
            if len(text.strip()) < 8:
                continue

            tokens = self.encode_fn(text)
            if len(tokens) > self.max_tokens_per_example:
                tokens = tokens[: self.max_tokens_per_example]

            for x, y in _segment_tokens_with_mask(
                tokens=tokens,
                prompt_len=0,
                encode_len=len(tokens),
                seq_len=self.seq_len,
                max_segments=self.max_segments_per_example,
            ):
                yield x, y

            n += 1
            if self.max_examples is not None and n >= self.max_examples:
                break


class OpenWebTextSegmentsDataset(IterableDataset):
    """
    Streams OpenWebText-like general text and yields next-token segments.
    No label masking (prompt_len = 0).
    """

    def __init__(
        self,
        encode_fn: Callable[[str], List[int]],
        seq_len: int,
        split: str = "train",
        max_examples: Optional[int] = None,
        max_tokens_per_example: int = 4096,
        max_segments_per_example: int = 8,
        dataset_name: str = "dylanebert/openwebtext",
    ):
        super().__init__()
        self.encode_fn = encode_fn
        self.seq_len = seq_len
        self.split = split
        self.max_examples = max_examples
        self.max_tokens_per_example = max_tokens_per_example
        self.max_segments_per_example = max_segments_per_example
        self.dataset_name = dataset_name

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("Run: pip install datasets")

        ds = load_dataset(self.dataset_name, split=self.split, streaming=True)
        n = 0
        for item in ds:
            text = item.get("text", None)  # type: ignore[union-attr]
            if not text:
                continue

            tokens = self.encode_fn(text)
            if len(tokens) > self.max_tokens_per_example:
                tokens = tokens[: self.max_tokens_per_example]

            # prompt_len=0 => never mask (no prompt region)
            for x, y in _segment_tokens_with_mask(
                tokens=tokens,
                prompt_len=0,
                encode_len=len(tokens),
                seq_len=self.seq_len,
                max_segments=self.max_segments_per_example,
            ):
                yield x, y

            n += 1
            if self.max_examples is not None and n >= self.max_examples:
                break


class AlpacaInstructionSegmentsDataset(IterableDataset):
    """
    Streams instruction/Q&A pairs (Alpaca-like) and yields next-token segments.
    Prompt tokens (question + instruction + separators) are ignored in the loss.
    """

    def __init__(
        self,
        encode_fn: Callable[[str], List[int]],
        seq_len: int,
        split: str = "train",
        max_examples: Optional[int] = None,
        max_tokens_per_example: int = 4096,
        max_segments_per_example: int = 8,
        dataset_name: str = "tatsu-lab/alpaca",
        instruction_ratio: float = 1.0,
    ):
        super().__init__()
        self.encode_fn = encode_fn
        self.seq_len = seq_len
        self.split = split
        self.max_examples = max_examples
        self.max_tokens_per_example = max_tokens_per_example
        self.max_segments_per_example = max_segments_per_example
        self.dataset_name = dataset_name
        self.instruction_ratio = instruction_ratio

    def _format_prompt(self, instruction: str, input_text: str) -> str:
        # Simple prompt template. Your tokenizer is SentencePiece, so newlines are fine.
        prompt = f"Instruction: {instruction}\n"
        if input_text:
            prompt += f"Input: {input_text}\n"
        prompt += "Answer: "
        return prompt

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("Run: pip install datasets")

        ds = load_dataset(self.dataset_name, split=self.split, streaming=True)
        n = 0
        for item in ds:
            instruction = item.get("instruction", "")  # type: ignore[union-attr]
            input_text = item.get("input", "")  # type: ignore[union-attr]
            output = item.get("output", "")  # type: ignore[union-attr]

            if not instruction or not output:
                continue

            prompt_text = self._format_prompt(str(instruction), str(input_text or ""))
            prompt_tokens = self.encode_fn(prompt_text)
            answer_tokens = self.encode_fn(str(output))

            tokens = prompt_tokens + answer_tokens
            if len(tokens) > self.max_tokens_per_example:
                tokens = tokens[: self.max_tokens_per_example]

            prompt_len = min(len(prompt_tokens), len(tokens))

            for x, y in _segment_tokens_with_mask(
                tokens=tokens,
                prompt_len=prompt_len,
                encode_len=len(tokens),
                seq_len=self.seq_len,
                max_segments=self.max_segments_per_example,
            ):
                yield x, y

            n += 1
            if self.max_examples is not None and n >= self.max_examples:
                break


class MixedGeneralInstructionDataset(IterableDataset):
    """
    Mixes general text + instruction data and yields (x, labels) segments.

    instruction_ratio:
      Probability of drawing the next example from the instruction dataset.
    """

    def __init__(
        self,
        encode_fn: Callable[[str], List[int]],
        seq_len: int,
        split: str = "train",
        instruction_ratio: float = 0.5,
        general_dataset_name: str = "dylanebert/openwebtext",
        instruction_dataset_name: str = "tatsu-lab/alpaca",
        max_tokens_per_example: int = 4096,
        max_segments_per_example: int = 8,
    ):
        super().__init__()
        self.encode_fn = encode_fn
        self.seq_len = seq_len
        self.split = split
        self.instruction_ratio = instruction_ratio
        self.general_dataset_name = general_dataset_name
        self.instruction_dataset_name = instruction_dataset_name
        self.max_tokens_per_example = max_tokens_per_example
        self.max_segments_per_example = max_segments_per_example

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # Build two streaming datasets and interleave examples at the segment-level.
        general_ds = OpenWebTextSegmentsDataset(
            encode_fn=self.encode_fn,
            seq_len=self.seq_len,
            split=self.split,
            dataset_name=self.general_dataset_name,
            max_tokens_per_example=self.max_tokens_per_example,
            max_segments_per_example=self.max_segments_per_example,
        )
        instruction_ds = AlpacaInstructionSegmentsDataset(
            encode_fn=self.encode_fn,
            seq_len=self.seq_len,
            split=self.split,
            dataset_name=self.instruction_dataset_name,
            max_tokens_per_example=self.max_tokens_per_example,
            max_segments_per_example=self.max_segments_per_example,
        )

        general_it = iter(general_ds)
        instruction_it = iter(instruction_ds)

        pending: List[Tuple[torch.Tensor, torch.Tensor]] = []
        while True:
            if pending:
                yield pending.pop(0)
                continue

            # Pick which dataset to draw the next example from
            if random.random() < self.instruction_ratio:
                try:
                    # Pull one segment at a time; pending is filled by taking a few segments.
                    for _ in range(self.max_segments_per_example):
                        pending.append(next(instruction_it))
                except StopIteration:
                    continue
            else:
                try:
                    for _ in range(self.max_segments_per_example):
                        pending.append(next(general_it))
                except StopIteration:
                    continue


class MixedConversationDataset(IterableDataset):
    """
    Conversation-first curriculum:
      - **TinyStories**: simple, clear narrative English (easy to read aloud).
      - **UltraChat** (multi-turn): natural chat / Q&A turns (parquet Hub; replaces legacy DailyDialog).
      - **Alpaca**: instruction + answer with masked prompt (Q&A / assistant style).

    Weights are normalized to sum to 1. Default emphasizes story + dialogue together.
    """

    def __init__(
        self,
        encode_fn: Callable[[str], List[int]],
        seq_len: int,
        split: str = "train",
        story_ratio: float = 0.35,
        dialogue_ratio: float = 0.35,
        instruction_ratio: float = 0.30,
        alpaca_dataset_name: str = "tatsu-lab/alpaca",
        max_tokens_per_example: int = 4096,
        max_segments_per_example: int = 8,
        chat_split: str = "train_sft",
    ):
        super().__init__()
        self.encode_fn = encode_fn
        self.seq_len = seq_len
        self.split = split
        self.chat_split = chat_split
        tot = float(story_ratio + dialogue_ratio + instruction_ratio)
        if tot <= 0:
            raise ValueError("story_ratio + dialogue_ratio + instruction_ratio must be > 0")
        self.w_story = story_ratio / tot
        self.w_dialog = dialogue_ratio / tot
        self.w_inst = instruction_ratio / tot
        self.alpaca_dataset_name = alpaca_dataset_name
        self.max_tokens_per_example = max_tokens_per_example
        self.max_segments_per_example = max_segments_per_example

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        ts = TinyStoriesSegmentsDataset(
            encode_fn=self.encode_fn,
            seq_len=self.seq_len,
            split=self.split,
            dataset_name="roneneldan/TinyStories",
            max_tokens_per_example=self.max_tokens_per_example,
            max_segments_per_example=self.max_segments_per_example,
        )
        dd = MultiTurnChatSegmentsDataset(
            encode_fn=self.encode_fn,
            seq_len=self.seq_len,
            split=self.chat_split,
            dataset_name="HuggingFaceH4/ultrachat_200k",
            max_tokens_per_example=self.max_tokens_per_example,
            max_segments_per_example=self.max_segments_per_example,
        )
        alp = AlpacaInstructionSegmentsDataset(
            encode_fn=self.encode_fn,
            seq_len=self.seq_len,
            split=self.split,
            dataset_name=self.alpaca_dataset_name,
            max_tokens_per_example=self.max_tokens_per_example,
            max_segments_per_example=self.max_segments_per_example,
        )

        ts_it = iter(ts)
        dd_it = iter(dd)
        alp_it = iter(alp)

        while True:
            r = random.random()
            if r < self.w_story:
                try:
                    yield next(ts_it)
                except StopIteration:
                    ts_it = iter(ts)
            elif r < self.w_story + self.w_dialog:
                try:
                    yield next(dd_it)
                except StopIteration:
                    dd_it = iter(dd)
            else:
                try:
                    yield next(alp_it)
                except StopIteration:
                    alp_it = iter(alp)

