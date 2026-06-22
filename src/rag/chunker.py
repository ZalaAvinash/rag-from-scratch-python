"""Split long text into overlapping chunks suitable for embedding.

Why chunk?
    Embedding models have a max input length (typically 256-512 tokens).
    Even when they don't, smaller chunks retrieve more precisely — a 500-token
    question is more likely to match a 200-token passage than a 5000-token one.

Why overlap?
    Without overlap, a sentence that spans a chunk boundary gets cut in half and
    neither chunk contains it cleanly. Overlap keeps continuity at boundaries.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A piece of text with its position in the source document."""
    text: str
    index: int            # ordinal position in the chunk stream
    char_start: int       # offset in the source document
    char_end: int         # offset in the source document


def _normalize(text: str) -> str:
    """Collapse runs of whitespace and strip — keeps char counts meaningful."""
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """Split text into overlapping windows of approximately `chunk_size` characters.

    Algorithm:
        Slide a window of size `chunk_size` across the text, advancing by
        (chunk_size - chunk_overlap) each step. Chunks are normalized (whitespace
        collapsed) for cleaner embedding, but the returned Chunk records the
        original offsets so callers can map back to the source.

    Args:
        text: Source text. Will be normalized before chunking.
        chunk_size: Target size of each chunk in characters (after normalization).
        chunk_overlap: Number of characters shared between adjacent chunks.

    Returns:
        List of Chunk objects. Empty if input is empty/whitespace.

    Raises:
        ValueError: If chunk_size <= 0, chunk_overlap < 0, or overlap >= size.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
        )

    normalized = _normalize(text)
    if not normalized:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[Chunk] = []
    i = 0
    idx = 0
    n = len(normalized)

    while i < n:
        piece = normalized[i : i + chunk_size]
        # Find the original-text char range for this normalized slice
        char_start = _normalized_to_original_offset(text, i)
        char_end = _normalized_to_original_offset(text, min(i + chunk_size, n))
        chunks.append(Chunk(text=piece, index=idx, char_start=char_start, char_end=char_end))
        idx += 1
        i += step

    return chunks


def _normalized_to_original_offset(text: str, normalized_offset: int) -> int:
    """Map an offset in the normalized text back to an offset in the original.

    Used so callers (UI, source highlighters) can point at the original PDF
    location even though we collapsed whitespace for chunking.
    """
    if normalized_offset <= 0:
        return 0
    out_i = 0
    norm_i = 0
    in_ws = False
    for ch in text:
        if ch.isspace():
            if not in_ws:
                # We just emitted a single space into the normalized stream
                if norm_i >= normalized_offset:
                    return out_i
                norm_i += 1
                in_ws = True
        else:
            if in_ws:
                in_ws = False
            if norm_i >= normalized_offset:
                return out_i
            norm_i += 1
        out_i += 1
    return out_i
