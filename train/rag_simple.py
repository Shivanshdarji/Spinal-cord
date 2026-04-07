import os
import re
from pathlib import Path
from typing import List, Tuple


def _normalize_words(text: str) -> List[str]:
    # Lowercase + keep only alphanumerics/underscores.
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def chunk_words(text: str, *, chunk_words: int = 120, overlap_words: int = 20) -> List[str]:
    words = _normalize_words(text)
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_words)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap_words)
    return chunks


def load_corpus_chunks(rag_dir: str, *, chunk_words: int = 120, overlap_words: int = 20) -> List[str]:
    p = Path(rag_dir)
    if not p.exists() or not p.is_dir():
        raise ValueError(f"RAG dir not found or not a directory: {rag_dir}")

    all_chunks: List[str] = []
    for file in p.glob("**/*.txt"):
        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        all_chunks.extend(chunk_words(text, chunk_words=chunk_words, overlap_words=overlap_words))
    return all_chunks


def retrieve_top_k(query: str, chunks: List[str], *, k: int = 3) -> List[Tuple[str, int]]:
    """
    Very simple overlap-based retrieval:
      score(chunk) = count of shared words with query.
    """
    q_words = _normalize_words(query)
    if not q_words:
        return []
    q_set = set(q_words)

    scored: List[Tuple[str, int]] = []
    for c in chunks:
        c_words = _normalize_words(c)
        score = sum(1 for w in c_words if w in q_set)
        if score > 0:
            scored.append((c, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def build_rag_prompt(user_prompt: str, retrieved_chunks: List[str]) -> str:
    if not retrieved_chunks:
        return user_prompt

    context = "\n\n".join(retrieved_chunks)
    return (
        "Context:\n"
        f"{context}\n\n"
        "User:\n"
        f"{user_prompt}\n\n"
        "Answer:\n"
    )

