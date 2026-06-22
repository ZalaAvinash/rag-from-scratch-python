"""Tests for the chunker — the only piece with pure-function logic that's easy
to test without hitting Ollama or Chroma.
"""
import pytest

from rag.chunker import Chunk, chunk_text


def make_text(words: int) -> str:
    return " ".join(f"word{i}" for i in range(words))


def test_empty_text_returns_empty_list():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_text_returns_single_chunk():
    text = "hello world"
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert "hello" in chunks[0].text
    assert "world" in chunks[0].text


def test_long_text_produces_multiple_chunks():
    text = make_text(500)
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    # All chunks should be non-empty
    assert all(c.text for c in chunks)
    # Indexes should be sequential starting at 0
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunks_overlap():
    text = make_text(300)
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=30)
    # Chunk 0 and chunk 1 should share ~30 chars (after whitespace normalization)
    end_of_first = chunks[0].text[-30:].split()
    start_of_second = chunks[1].text[:60].split()
    # At least some words from end of chunk 0 should appear in chunk 1
    assert any(w in start_of_second for w in end_of_first)


def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=0)


def test_negative_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=100, chunk_overlap=-5)


def test_overlap_larger_than_size_raises():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=50, chunk_overlap=60)


def test_chunk_offsets_advance():
    text = make_text(500)
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    for c in chunks:
        assert c.char_start <= c.char_end
    # Each subsequent chunk starts at or after the previous one's start
    for prev, curr in zip(chunks, chunks[1:]):
        assert curr.char_start >= prev.char_start


def test_whitespace_is_normalized_in_output():
    chunks = chunk_text("hello   world\n\n\nfoo\t\tbar", chunk_size=200, chunk_overlap=20)
    assert len(chunks) == 1
    # Whitespace should be collapsed to single spaces
    assert chunks[0].text == "hello world foo bar"
